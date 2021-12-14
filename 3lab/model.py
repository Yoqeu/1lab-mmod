import numpy as np
from matplotlib import pyplot as plt

from query import *
from terminal import *


class Model:
    def __init__(self, m, lambda_, mu, nu, r, max_queries_processed):
        self._m = m
        self._lambda = lambda_
        self._mu = mu
        self._nu = nu
        self._r = r
        self._max_queries_processed = max_queries_processed

        self._terminal_busy = False
        self._terminal_broken = False
        self._break_timepoint_determined = False
        self._timeline = []
        self._queue = []

        self._queries_processed = 0
        self._queries_dropped = 0

        self._final_state_durations = [0] * (2 * self._m + 2)
        self._state_durations = []
        self._last_state = 0
        self._last_state_change_timepoint = 0

    def start(self):
        query_gen = self._generate(self._lambda)
        service_gen = self._generate(self._mu)
        break_gen = self._generate(self._nu)
        repair_gen = self._generate(self._r)

        first_query_timepoint = next(query_gen)
        self._timeline.append(QueryArrived(first_query_timepoint))

        for _ in range(self._max_queries_processed - 1):
            timepoint = self._timeline[len(self._timeline) - 1].timepoint + next(query_gen)
            self._timeline.append(QueryArrived(timepoint))

        for event in self._timeline:
            if isinstance(event, QueryArrived):
                if not self._terminal_busy and not self._terminal_broken:
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessed(event.timepoint, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreak(break_timepoint))
                        self._break_timepoint_determined = True

                    self._record_state(event.timepoint)
                else:
                    if len(self._queue) < self._m:
                        self._queue.append(event.timepoint)
                        self._record_state(event.timepoint)
                    else:
                        # drop
                        self._queries_dropped += 1

            if isinstance(event, QueryProcessed):
                self._queries_processed += 1
                self._terminal_busy = False

                if len(self._queue) != 0:
                    waiting_start = self._queue.pop(0)
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessed(waiting_start, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreak(break_timepoint))
                        self._break_timepoint_determined = True

                self._record_state(event.timepoint)

            if isinstance(event, TerminalBreak) and event != self._timeline[-1]:
                self._break_timepoint_determined = False
                self._terminal_broken = True

                removed_event = self._remove_nearest_query_processed_event(event)

                repair_timepoint = event.timepoint + next(repair_gen)
                repair_event = TerminalRepair(repair_timepoint)
                self._insert(repair_event)

                if len(self._queue) < self._m:
                    self._queue.append(removed_event.start_timepoint if removed_event != None else event.timepoint)
                else:
                    # drop
                    self._queries_dropped += 1

                self._record_state(event.timepoint)

            if isinstance(event, TerminalRepair):
                self._terminal_broken = False
                self._terminal_busy = False

                if len(self._queue) != 0:
                    waiting_start = self._queue.pop(0)
                    self._terminal_busy = True
                    query_processed_timepoint = event.timepoint + next(service_gen)
                    query_processed_event = QueryProcessed(waiting_start, query_processed_timepoint)
                    self._insert(query_processed_event)

                    if not self._break_timepoint_determined:
                        break_timepoint = event.timepoint + next(break_gen)
                        self._insert(TerminalBreak(break_timepoint))
                        self._break_timepoint_determined = True

                self._record_state(event.timepoint)

    def _generate(self, param):
        while True:
            yield np.random.exponential(1 / param)

    def _insert(self, event):
        for i in range(1, len(self._timeline)):
            if self._timeline[i - 1].timepoint < event.timepoint < self._timeline[i].timepoint:
                self._timeline.insert(i, event)
                break
        else:
            self._timeline.append(event)

    def _remove_nearest_query_processed_event(self, terminal_break_event):
        break_event_index = self._timeline.index(terminal_break_event)

        for i in range(break_event_index, len(self._timeline)):
            if isinstance(self._timeline[i], QueryProcessed):
                return self._timeline.pop(i)

    def _record_state(self, timepoint):
        time_delta = timepoint - self._last_state_change_timepoint
        self._final_state_durations[self._last_state] += time_delta

        self._last_state_change_timepoint = timepoint

        if not self._terminal_broken:
            self._last_state = int(self._terminal_busy) + len(self._queue)
        else:
            self._last_state = 1 + self._m + len(self._queue)

        self._state_durations.append((timepoint, self._final_state_durations.copy()))

    def show_stationary_stats(self):
        x = [timepoint for timepoint, _ in self._state_durations]

        for i in range(len(self._final_state_durations)):
            y = [durations[i] / timepoint for timepoint, durations in self._state_durations]
            plt.plot(x, y)

        plt.show()

    def show_stats(self):
        print('Queries processed:', self._queries_processed)
        print('Quries dropped:', self._queries_dropped)

        return [duration / self._timeline[-1].timepoint for duration in self._final_state_durations]
