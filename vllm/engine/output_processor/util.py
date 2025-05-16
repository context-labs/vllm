# SPDX-License-Identifier: Apache-2.0

from typing import List
from typing import Sequence as GenericSequence
from typing import cast

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import CompletionSequenceGroupOutput, SequenceGroupOutput


def create_output_by_sequence_group(
        outputs: GenericSequence[SamplerOutput],
        num_seq_groups: int,
        include_hidden_states: bool) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[CompletionSequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    print("In create_output_by_sequence_group")
    for step in outputs:
        sequence_group_output: CompletionSequenceGroupOutput
        start_idx = 0
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output)
            print("include_hidden_states:", include_hidden_states, "step.hidden_states is not None:", step.hidden_states is not None)
            if include_hidden_states and step.hidden_states is not None:
                print("In create_output_by_sequence_group, found hidden states.")
                num_seqs = len(sequence_group_output.samples)
                end_idx = start_idx + num_seqs
                sequence_group_output.hidden_states = (
                    step.hidden_states[start_idx:end_idx, :])
                start_idx = end_idx

    # Cast to the more generic type that CompletionSequenceGroupOutput
    # inherits from.
    return cast(List[List[SequenceGroupOutput]], output_by_sequence_group)
