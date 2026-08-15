from collections import deque
import cupy as cp


class StreamRingPool:
    __slots__ = ("worker_streams", "null_stream")
    '''
    https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Stream.html
    '''

    def __init__(self, num_streams=4):
        '''
        Stream pool designed for CPU-side managment. Uses a cirular buffer
        to evenly distribute the workload across the streams.
        There's no need to trace which streams are "in use" because they will
        be put back inside .worker_streams before it actually finish the queued
        computation.
        '''
        self.worker_streams = deque([cp.cuda.Stream(non_blocking=True) for _ in range(num_streams)])
        # stream di default
        self.null_stream = cp.cuda.Stream.null

    def get_stream(self) -> cp.cuda.Stream:
        return self.worker_streams.popleft()

    def _set_num_streams(self, n_streams: int):
        '''
        This function should only be called by set_max_async_streams()
        '''
        # mi assicuro non abbiano operazioni pending o in esecuzione
        for worker in self.worker_streams:
            worker.synchronize()

        while len(self.worker_streams) != n_streams:
            if len(self.worker_streams) > n_streams:
                self.worker_streams.popleft()
            else:
                self.worker_streams.append(cp.cuda.Stream(non_blocking=True))

    def release_stream(self, stream: cp.cuda.Stream) -> None:
        # Rimette la stream in fondo alla coda per il prossimo utilizzo
        self.worker_streams.append(stream)

    def can_parallelize(self):
        # per ora naif, in teoria dovrei considerare anche se ci sono
        # le condizioni hardware per farlo
        return len(self.worker_streams) > 0





class EventPool:
    '''
    Cio' che conta degli eventi e' quando gli registri, non quando
    li crei. Si puo' quindi creare un numero di eventi e poi riciclare
    sempre le stesse istanze (come le streams) senza sovraccaricare
    la CPU e il garbage collector
    '''
    __slots__ = ("available_events", "allocated_events", "workers_done_events")

    def __init__(self):
        self.available_events = []
        self.allocated_events = []
        self.workers_done_events = []

    def get_event(self) -> cp.cuda.Event:
        # Preleva dalla lista di quelli liberi o ne crea uno nuovo
        if self.available_events:
            event = self.available_events.pop()
        else:
            event = cp.cuda.Event()
            
        self.allocated_events.append(event)
        return event

    def register_worker_done_event(self, worker_stream) -> None:
        work_done_event = event_pool.get_event()
        worker_stream.record(work_done_event)
        self.workers_done_events.append(work_done_event)

    def wait_workers(self, null_stream) -> None:
        for event in self.workers_done_events:
            null_stream.wait_event(event)
        self.workers_done_events.clear()

    def release_events(self) -> None:
        '''
        Da chiamare a FINE BACKWARD.
        Sposta tutti gli eventi usati nella lista di quelli disponibili.
        '''
        if self.allocated_events:
            self.available_events.extend(self.allocated_events)
            self.allocated_events.clear()


#stream_pool = StreamPool()
stream_ring_pool = StreamRingPool()
event_pool = EventPool()
# StreamRingPool

