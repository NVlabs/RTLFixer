from vcdvcd import VCDVCD, binary_string_to_hex, StreamParserCallbacks
import math
import io
import pandas as pd
from typing import List 


class CustomCallback(StreamParserCallbacks):
    def __init__(self, printIds={}, lines=20, offset=0):
        self._printIdx = printIds
        self._references_to_widths = {}
        self.lines=20
        self.counter=0
        self.offset=offset
        
    def enddefinitions(
        self,
        vcd,
        signals,
        cur_sig_vals
    ):
        vcd.io = io.StringIO()
        self._printIdx = self._printIdx if self._printIdx else {i: i.split('.')[-1] for i in vcd.signals}
        
        if signals:
            self._print_dumps_refs = signals
        else:
            self._print_dumps_refs = sorted(vcd.data[i].references[0] for i in cur_sig_vals.keys())

        for i, ref in enumerate(self._print_dumps_refs, 1):
            if i == 0:
                i = 1
            identifier_code = vcd.references_to_ids[ref]
            size = int(vcd.data[identifier_code].size)
            width = max(((size // 4)), int(math.floor(math.log10(i))) + 1)
            self._references_to_widths[ref] = width

        to_print = '// {0:<16}'.format('time')
        for ref in vcd.signals:
            string = '{0:>{1}s}'.format(self._printIdx[ref], self._references_to_widths.get(ref, 1))
            to_print += '{0:<16}'.format(string)
            
        print(to_print, file=vcd.io)

        
    def time(
        self,
        vcd,
        time,
        cur_sig_vals
    ):
        self.counter += 1
        

        if self.counter > self.offset + self.lines or self.counter < self.offset:
            return

        if (vcd.signal_changed):
            ss = []
            ss.append('// {0:<16}'.format(str(time)+'ns'))
            for ref in self._printIdx:
                identifier_code = vcd.references_to_ids[ref]
                value = cur_sig_vals[identifier_code]
                string = '{0:>{1}s}'.format(
                    binary_string_to_hex(value),
                    self._references_to_widths.get(ref, 1))
                ss.append( '{0:<16}'.format(string) )
            print(''.join(ss), file=vcd.io)

            
            
            
            
            
def tabular_via_callback(vcd_path, offset: int, mismatch_columns: List[str], window_size: int = 5):
    vcd = VCDVCD(vcd_path, callbacks=CustomCallback(offset=offset, lines=window_size), store_tvs=False, only_sigs=False)
    tabular_text = vcd.io.getvalue()
    return tabular_text

def tabular_via_dataframe(vcd_path, offset: int, mismatch_columns: List[str], window_size: int = 5):
#     from scipy.sparse import csc_matrix
    import numpy as np

    vcd = VCDVCD(vcd_path)
    n_row = vcd.endtime + 1
    n_col = len(vcd.signals)
    matrix = np.full((n_row, n_col), np.nan, dtype=float)
    for e, ref in enumerate(vcd.signals):
        symbol = vcd.references_to_ids[ref]
        for ts, signal in vcd.data[symbol].tv:
            try:
                matrix[ts, e] = int(signal) if signal.isdigit() else -999
            except:
                matrix[ts, e] = -999

    df = pd.DataFrame(matrix, columns=[i.split(".")[-1] for i in vcd.signals]).dropna(subset='clk')
    df = df.fillna(method='ffill')

    mismatch_columns = [i for i in df.columns if any(j in i for j in mismatch_columns)]
    first_row = df.loc[0: 1][mismatch_columns]
    tail_rows = df.loc[1: offset+1][mismatch_columns].drop_duplicates(keep='first')
    df = pd.concat([first_row, tail_rows])[-window_size:]
    df = df.astype(int).astype(str).applymap(lambda x: binary_string_to_hex(x) if x != -999 else 'x')
    df.index.names = ['time(ns)']
    return df.to_string(header=True, index=True)
