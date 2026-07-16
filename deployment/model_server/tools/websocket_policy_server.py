# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

import asyncio
import logging
import time
import traceback

import websockets.asyncio.server
import websockets.frames

# from openpi_client import base_policy as _base_policy
from . import msgpack_numpy


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Supports optional request batching for infer/predict_action to improve throughput
    when many workers connect to a single server.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: int = -1,  # Idle timeout in seconds, -1 means never auto-close
        metadata: dict | None = None,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 30.0,
    ) -> None:
        self._policy = policy  #
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._max_batch_size = max_batch_size
        self._batch_timeout_ms = batch_timeout_ms
        self._infer_queue: asyncio.Queue = asyncio.Queue()
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            batch_task = asyncio.create_task(self._batch_processor())
            try:
                if self._idle_timeout > 0:
                    await self._idle_watchdog(server)
                else:
                    await server.serve_forever()
            finally:
                batch_task.cancel()
                try:
                    await batch_task
                except asyncio.CancelledError:
                    pass

    async def _batch_processor(self):
        """Collect infer requests into batches and process them together."""
        while True:
            # Wait for at least one request
            first_item = await self._infer_queue.get()
            batch = [first_item]

            # Collect more requests until batch is full or timeout expires
            deadline = time.time() + self._batch_timeout_ms / 1000.0
            while len(batch) < self._max_batch_size and time.time() < deadline:
                try:
                    item = self._infer_queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    try:
                        # Wait a tiny bit for more requests without blocking forever
                        remaining = deadline - time.time()
                        if remaining > 0:
                            item = await asyncio.wait_for(
                                self._infer_queue.get(), timeout=max(remaining, 0.001)
                            )
                            batch.append(item)
                    except asyncio.TimeoutError:
                        break

            await self._process_batch(batch)

    async def _process_batch(self, batch):
        """Run a batch of infer requests through the policy and return results."""
        futures = []
        examples = []
        kwargs_common = {}

        for future, msg, payload in batch:
            futures.append(future)
            ex_list = payload.get("examples")
            if isinstance(ex_list, list) and ex_list:
                examples.extend(ex_list)
            elif isinstance(payload, dict):
                # Fallback: treat payload itself as a single example if it has image/lang
                examples.append(payload)

            # Use kwargs from the first request as the common kwargs
            if not kwargs_common and isinstance(payload, dict):
                for k, v in payload.items():
                    if k != "examples":
                        kwargs_common[k] = v

        if not examples:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(ValueError("No examples in batch"))
            return

        try:
            output_dict = self._policy.predict_action(examples=examples, **kwargs_common)
            actions = output_dict["actions"]  # (B, T, D)
            timings = output_dict.get("timings", {})

            if len(actions) < len(futures):
                raise RuntimeError(
                    f"Batch output actions count ({len(actions)}) less than requests ({len(futures)})"
                )

            for i, fut in enumerate(futures):
                if fut.done():
                    continue
                ret = {
                    "status": "ok",
                    "ok": True,
                    "type": "inference_result",
                    "request_id": batch[i][1].get("request_id", "default"),
                    "data": {
                        "actions": actions[i : i + 1],
                        "timings": timings,
                    },
                }
                fut.set_result(ret)
        except Exception as e:
            logging.exception("Batch inference error")
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    async def _idle_watchdog(self, server):
        """Monitor idle time and shut down the server on timeout."""
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                msg = msgpack_numpy.unpackb(await websocket.recv())
                self._last_active = time.time()  # Refresh active time on each received message

                mtype = msg.get("type", "infer")
                if mtype in ("infer", "predict_action"):
                    payload = msg.get("payload", msg)
                    future = asyncio.get_event_loop().create_future()
                    await self._infer_queue.put((future, msg, payload))
                    ret = await future
                else:
                    ret = self._route_message(msg)

                await websocket.send(packer.pack(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    # route logic: recognize request from client
    def _route_message(self, msg: dict) -> dict:
        """
        Route rules (fault-tolerant):
        - Supports messages of form:
            {"type": "ping|init|infer|reset", "request_id": "...", "payload": {...}}
          or a flat dict (will be treated as payload).
        - Does NOT raise inside this function: all exceptions are caught and encoded in response.
        """
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "infer")  # default = infer
        payload = msg.get("payload", msg)  # when no explicit payload, treat top-level as payload

        # ping
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}

        # infer --> framework.predict_action
        elif mtype == "infer" or mtype == "predict_action":
            # Basic payload sanity
            if not isinstance(payload, dict):
                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {"message": "Payload must be a dict", "payload_type": str(type(payload))},
                }
            try:
                output_dict = self._policy.predict_action(**payload)
            except Exception as e:
                logging.exception("Policy inference error (request_id=%s)", req_id)
                logging.exception(e)

                return {
                    "status": "error",
                    "ok": False,
                    "type": "inference_result",
                    "request_id": req_id,
                    "error": {
                        "message": str(e),
                    },
                }
            data = output_dict
            return {
                "status": "ok",
                "ok": True,
                "type": "inference_result",
                "request_id": req_id,
                "data": data,
            }

        # unknow request type
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()
    raise NotImplementedError("This module is not intended to be run directly.")
#
#  Instead, it should be imported and used in a server context.
