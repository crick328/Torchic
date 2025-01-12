import cupy as cp
from torchic.settings import CONFIG



class StreamPool:
    __slots__ = ("available_streams", "streams_in_use", "null_stream", "default_stream")

    def __init__(self):
        self.available_streams = set()
        self.streams_in_use = set()
        self.null_stream = cp.cuda.Stream.null
        self.default_stream = None

        self.set_default_stream(CONFIG["cuda"]["use_null"])
        self.set_available_streams(CONFIG["cuda"]["max_async_streams"])

    # faccio cosi' in modo da poterla cambiare come setting in runtime
    def set_default_stream(self, use_null: bool):
        self.default_stream = self.null_stream if use_null else cp.cuda.Stream(non_blocking=True)

    def set_available_streams(self, async_streams: int):
        self.sync_all()
        
        while len(self.available_streams) != async_streams:
            if len(self.available_streams) > async_streams:
                stream = self.available_streams.pop()
                stream.release()
            else:
                self.available_streams.add(cp.cuda.Stream(non_blocking=True))

    def get_stream(self) -> cp.cuda.Stream:
        if self.available_streams:
            stream = self.available_streams.pop()
            self.streams_in_use.add(stream)
            #print((len(self.streams_in_use)))

            if len(self.available_streams) == 0:
                print("ULTIMA STREAM!")
            return stream
        else:
            raise RuntimeError("No streams left!")
    
    def can_parallelize(self):
        # qua si aggiungera'
        # in valutazione delle restrizioni hardware
        return len(self.available_streams) > 0

    def release_stream(self, stream: cp.cuda.Stream) -> None:
        # NOTE: qua NON posso mettere un sync perche' e' un
        # callback e bloccherebbe tutto
        self.streams_in_use.remove(stream)
        self.available_streams.add(stream)
        
    def sync_critical_streams(self, events_dict):
        # la stream null NON e' in grado di aspettare
        # gli eventi, devono essere sincronizzati!
        if self.default_stream is self.null_stream:
            for stream_key in events_dict:
                events_dict[stream_key].synchronize()
        else:
            for stream_key in events_dict:
                self.default_stream.wait_event(events_dict[stream_key])
            # la stream non puo' aspettare il suo stesso
            # evento, quindi va sincronizzata separatamente
            # (prima o dopo fa differenza? Precedentemente lo
            # facevo prima)
            self.default_stream.synchronize()

        #self.null_stream.synchronize()

    def sync_all(self, _=None):
        # non ho bisogno di mettere la default tra quelle
        # in uso perche' se chiamo questa funzione vuol
        # dire che ha finito.
        # Per ora, il set e' vuoto al momento della chiamata
        for stream in list(self.streams_in_use):
            stream.synchronize()