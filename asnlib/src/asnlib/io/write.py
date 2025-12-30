from obspy import read
import os

def write_with_template(trace_template, new_data, out_dir, pattern, filename):
    fmt = read(trace_template)[0].copy()
    fmt.data = new_data.astype(fmt.data.dtype)

    os.makedirs(os.path.join(out_dir, pattern), exist_ok=True)
    fmt.write(os.path.join(out_dir, pattern,os.path.basename(filename)), format="MSEED")

    return