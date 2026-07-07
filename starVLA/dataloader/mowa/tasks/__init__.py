"""MoWA atomic-task label builders.

Importing this package registers all available builders with the global
registry in ``atomic_task_label_builder.py``.
"""

from __future__ import annotations

from starVLA.dataloader.mowa.tasks.close_blender_lid import CloseBlenderLidLabelBuilder
from starVLA.dataloader.mowa.tasks.close_toaster_oven_door import CloseToasterOvenDoorLabelBuilder
from starVLA.dataloader.mowa.tasks.open_stand_mixer_head import OpenStandMixerHeadLabelBuilder
from starVLA.dataloader.mowa.tasks.opencabinet import OpenCabinetLabelBuilder
from starVLA.dataloader.mowa.tasks.opendrawer import OpenDrawerLabelBuilder
from starVLA.dataloader.mowa.tasks.slide_dishwasher_rack import SlideDishwasherRackLabelBuilder
from starVLA.dataloader.mowa.tasks.turn_off_stove import TurnOffStoveLabelBuilder
from starVLA.dataloader.mowa.tasks.turn_on_electric_kettle import TurnOnElectricKettleLabelBuilder
from starVLA.dataloader.mowa.tasks.turn_on_sink_faucet import TurnOnSinkFaucetLabelBuilder

__all__ = [
    "CloseBlenderLidLabelBuilder",
    "CloseToasterOvenDoorLabelBuilder",
    "OpenCabinetLabelBuilder",
    "OpenDrawerLabelBuilder",
    "OpenStandMixerHeadLabelBuilder",
    "SlideDishwasherRackLabelBuilder",
    "TurnOffStoveLabelBuilder",
    "TurnOnElectricKettleLabelBuilder",
    "TurnOnSinkFaucetLabelBuilder",
]
