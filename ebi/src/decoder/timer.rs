use crate::time::SegmentedExecutionTimes;

pub trait ExecutionTimer {
    fn segmented_execution_times(&self) -> SegmentedExecutionTimes;
}
