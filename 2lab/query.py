class QueryArrived:
    def __init__(self, timepoint):
        self.timepoint = timepoint

    def __str__(self):
        return self.timepoint


class QueryProcessed:
    def __init__(self, start_timepoint, timepoint, terminal_no):
        self.start_timepoint = start_timepoint
        self.timepoint = timepoint
        self.terminal_no = terminal_no

    def __str__(self):
        return self.timepoint
