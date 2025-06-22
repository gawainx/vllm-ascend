from typing import Optional

from vllm.v1.core.sched.output import SchedulerOutput


class PrefixSchedulerOutput(SchedulerOutput):
    block_copy_mapping: Optional[dict[int, int]] = None
    copy_length: list[int] = []
