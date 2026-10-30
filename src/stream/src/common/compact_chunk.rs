// Copyright 2024 RisingWave Labs
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use itertools::Itertools;
use risingwave_common::array::stream_chunk::StreamChunkMut;
use risingwave_common::array::stream_record::Record;
use risingwave_common::array::{Op, StreamChunk};
use risingwave_common::row::RowExt;
use risingwave_common::types::DataType;

pub use super::change_buffer::InconsistencyBehavior;
use crate::common::change_buffer::output_kind::{RETRACT, UPSERT};
use crate::common::change_buffer::{ChangeBuffer, OutputKind};

/// A helper to remove unnecessary changes in the stream chunks based on the key.
///
/// By default, the input is treated as a retract stream. Use [`Self::with_upsert_input`] if the
/// input is an upsert stream, where an `Insert` may overwrite a row delivered earlier and a
/// `Delete` is a tombstone that must not be cancelled by a preceding `Insert`. The output kind
/// only controls the representation of updates and does not affect how the input is interpreted.
pub struct StreamChunkCompactor {
    chunks: Vec<StreamChunk>,
    key: Vec<usize>,
    upsert_input: bool,
}

impl StreamChunkCompactor {
    pub fn new(key: Vec<usize>, chunks: Vec<StreamChunk>) -> Self {
        Self {
            chunks,
            key,
            upsert_input: false,
        }
    }

    /// Set whether the input chunks are from an upsert stream.
    pub fn with_upsert_input(mut self, upsert_input: bool) -> Self {
        self.upsert_input = upsert_input;
        self
    }

    pub fn into_inner(self) -> (Vec<StreamChunk>, Vec<usize>) {
        (self.chunks, self.key)
    }

    /// Remove unnecessary changes in the given chunks, by modifying the visibility and ops in place.
    pub fn into_compacted_chunks_inline<const KIND: OutputKind>(
        self,
        ib: InconsistencyBehavior,
    ) -> Vec<StreamChunk> {
        let upsert_input = self.upsert_input;
        let (chunks, key_indices) = self.into_inner();

        let estimate_size = chunks.iter().map(|c| c.cardinality()).sum();
        let mut cb = ChangeBuffer::with_capacity(estimate_size)
            .with_inconsistency_behavior(ib)
            .with_upsert_input(upsert_input);

        let mut chunks = chunks.into_iter().map(StreamChunkMut::from).collect_vec();
        for chunk in &mut chunks {
            for (row, mut op_row) in chunk.to_rows_mut() {
                let op = op_row.op().normalize_update();
                let key = row.project(&key_indices);
                // Make all rows invisible first.
                op_row.set_vis(false);
                op_row.set_op(op);
                cb.apply_op_row(op, key, op_row);
            }
        }

        // For the rows that survive compaction, make them visible.
        for record in cb.into_records() {
            match record {
                Record::Insert { mut new_row } => new_row.set_vis(true),
                Record::Delete { mut old_row } => old_row.set_vis(true),
                Record::Update {
                    mut old_row,
                    mut new_row,
                } => {
                    match KIND {
                        // For upsert output, we only keep the new row with normalized op (`Insert`).
                        UPSERT => new_row.set_vis(true),
                        // For retract output, we keep both the old and new row, and rewrite the ops
                        // to `U-` and `U+` if they are adjacent.
                        RETRACT => {
                            old_row.set_vis(true);
                            new_row.set_vis(true);
                            if old_row.same_chunk(&new_row)
                                && old_row.index() + 1 == new_row.index()
                            {
                                old_row.set_op(Op::UpdateDelete);
                                new_row.set_op(Op::UpdateInsert);
                            }
                        }
                    }
                }
            }
        }

        chunks.into_iter().map(|c| c.into()).collect()
    }

    /// Remove unnecessary changes in the given chunks, by filtering them out and constructing new
    /// chunks, with the given chunk size.
    pub fn into_compacted_chunks_reconstructed<const KIND: OutputKind>(
        self,
        chunk_size: usize,
        data_types: Vec<DataType>,
        ib: InconsistencyBehavior,
    ) -> Vec<StreamChunk> {
        let upsert_input = self.upsert_input;
        let (chunks, key_indices) = self.into_inner();

        let estimate_size = chunks.iter().map(|c| c.cardinality()).sum();
        let mut cb = ChangeBuffer::with_capacity(estimate_size)
            .with_inconsistency_behavior(ib)
            .with_upsert_input(upsert_input);

        for chunk in &chunks {
            for record in chunk.records() {
                cb.apply_record(record, |&row| row.project(&key_indices));
            }
        }

        cb.into_chunks::<KIND>(data_types, chunk_size)
    }
}

/// Remove unnecessary changes in the given chunk, by modifying the visibility and ops in place.
///
/// This is the same as [`StreamChunkCompactor::into_compacted_chunks_inline`] with only one chunk.
pub fn compact_chunk_inline<const KIND: OutputKind>(
    stream_chunk: StreamChunk,
    key_indices: &[usize],
    upsert_input: bool,
    ib: InconsistencyBehavior,
) -> StreamChunk {
    Itertools::exactly_one(
        StreamChunkCompactor::new(key_indices.to_vec(), vec![stream_chunk])
            .with_upsert_input(upsert_input)
            .into_compacted_chunks_inline::<KIND>(ib)
            .into_iter(),
    )
    .unwrap_or_else(|_| unreachable!("should have exactly one chunk in the output"))
}

#[cfg(test)]
mod tests {
    use risingwave_common::test_prelude::StreamChunkTestExt;

    use super::*;

    #[test]
    fn test_compact_chunk_inline_upsert() {
        test_compact_chunk_inline::<UPSERT>();
    }

    #[test]
    fn test_compact_chunk_inline_retract() {
        test_compact_chunk_inline::<RETRACT>();
    }

    fn test_compact_chunk_inline<const KIND: OutputKind>() {
        let key = [0, 1];
        let chunks = vec![
            StreamChunk::from_pretty(
                " I I I
                - 1 1 1
                + 1 1 2
                + 2 5 7
                + 4 9 2
                - 2 5 7
                + 2 5 5
                - 6 6 9
                + 6 6 9
                - 9 9 1",
            ),
            StreamChunk::from_pretty(
                " I I I
                - 6 6 9
                + 9 9 9
                - 9 9 4
                + 2 2 2
                + 9 9 1",
            ),
        ];
        let compactor = StreamChunkCompactor::new(key.to_vec(), chunks);
        let mut iter = compactor
            .into_compacted_chunks_inline::<KIND>(InconsistencyBehavior::Panic)
            .into_iter();

        let chunk = iter.next().unwrap().compact_vis();
        let expected = match KIND {
            RETRACT => StreamChunk::from_pretty(
                " I I I
                U- 1 1 1
                U+ 1 1 2
                + 4 9 2
                + 2 5 5
                - 6 6 9",
            ),
            UPSERT => StreamChunk::from_pretty(
                " I I I
                + 1 1 2
                + 4 9 2
                + 2 5 5
                - 6 6 9",
            ),
        };
        assert_eq!(chunk, expected, "{}", chunk.to_pretty());

        let chunk = iter.next().unwrap().compact_vis();
        assert_eq!(
            chunk,
            StreamChunk::from_pretty(
                " I I I
                + 2 2 2",
            ),
            "{}",
            chunk.to_pretty()
        );

        assert_eq!(iter.next(), None);
    }

    /// For upsert input, the last operation of each key is kept, including the tombstone of a
    /// `Delete` following an `Insert`, which may have overwritten an existing downstream row.
    #[test]
    fn test_compact_chunk_upsert_input() {
        // Within one chunk.
        let chunk = StreamChunk::from_pretty(
            " I I
            + 1 20
            - 1 .
            + 2 5
            - 2 5
            + 2 6
            - 3 .
            - 3 .
            + 4 1
            + 4 2",
        );
        let expected = StreamChunk::from_pretty(
            " I I
            - 1 .
            + 2 6
            - 3 .
            + 4 2",
        );
        let compacted =
            compact_chunk_inline::<UPSERT>(chunk.clone(), &[0], true, InconsistencyBehavior::Panic);
        assert_eq!(compacted.compact_vis(), expected);
        let compacted =
            compact_chunk_inline::<RETRACT>(chunk, &[0], true, InconsistencyBehavior::Panic);
        assert_eq!(compacted.compact_vis(), expected);

        // Across chunks, and `Delete` followed by `Insert` of the same row must not be dropped as
        // a no-op update.
        let chunks = vec![
            StreamChunk::from_pretty(
                " I I
                + 1 20
                - 5 .",
            ),
            StreamChunk::from_pretty(
                " I I
                - 1 .
                + 5 7",
            ),
        ];
        let chunks = StreamChunkCompactor::new(vec![0], chunks)
            .with_upsert_input(true)
            .into_compacted_chunks_reconstructed::<UPSERT>(
                100,
                vec![DataType::Int64, DataType::Int64],
                InconsistencyBehavior::Panic,
            );
        assert_eq!(
            chunks,
            vec![StreamChunk::from_pretty(
                " I I
                - 1 .
                + 5 7",
            )]
        );

        let chunks = vec![StreamChunk::from_pretty(
            " I I
            - 1 7
            + 1 7",
        )];
        let chunks = StreamChunkCompactor::new(vec![0], chunks)
            .with_upsert_input(true)
            .into_compacted_chunks_inline::<UPSERT>(InconsistencyBehavior::Panic);
        assert_eq!(
            chunks.into_iter().next().unwrap().compact_vis(),
            StreamChunk::from_pretty(
                " I I
                + 1 7",
            )
        );
    }

    /// For retract input, an `Insert` cancelled by a `Delete` is still dropped.
    #[test]
    fn test_compact_chunk_retract_input_cancels() {
        let chunk = StreamChunk::from_pretty(
            " I I
            + 1 20
            - 1 20",
        );
        let compacted =
            compact_chunk_inline::<UPSERT>(chunk, &[0], false, InconsistencyBehavior::Panic);
        assert_eq!(compacted.cardinality(), 0);
    }

    #[test]
    fn test_compact_chunk_reconstructed_upsert() {
        test_compact_chunk_reconstructed::<UPSERT>();
    }

    #[test]
    fn test_compact_chunk_reconstructed_retract() {
        test_compact_chunk_reconstructed::<RETRACT>();
    }

    fn test_compact_chunk_reconstructed<const KIND: OutputKind>() {
        let key = [0, 1];
        let chunks = vec![
            StreamChunk::from_pretty(
                " I I I
            - 1 1 1
            + 1 1 2
            + 2 5 7
            + 4 9 2
            - 2 5 7
            + 2 5 5
            - 6 6 9
            + 6 6 9
            - 9 9 1",
            ),
            StreamChunk::from_pretty(
                " I I I
            - 6 6 9
            + 9 9 9
            - 9 9 4
            + 2 2 2
            + 9 9 1",
            ),
        ];
        let compactor = StreamChunkCompactor::new(key.to_vec(), chunks);

        let chunks = compactor.into_compacted_chunks_reconstructed::<KIND>(
            100,
            vec![DataType::Int64, DataType::Int64, DataType::Int64],
            InconsistencyBehavior::Panic,
        );
        let chunk = chunks.into_iter().next().unwrap();
        let expected = match KIND {
            RETRACT => StreamChunk::from_pretty(
                "  I I I
                 U- 1 1 1
                 U+ 1 1 2
                 + 4 9 2
                 + 2 5 5
                 - 6 6 9
                 + 2 2 2",
            ),
            UPSERT => StreamChunk::from_pretty(
                "  I I I
                 + 1 1 2
                 + 4 9 2
                 + 2 5 5
                 - 6 6 9
                 + 2 2 2",
            ),
        };
        assert_eq!(chunk, expected, "{}", chunk.to_pretty());
    }
}
