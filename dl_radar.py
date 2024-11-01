import nexradaws
# import pyart
from os import path
from pathlib import Path
from datetime import datetime as dt, timedelta
conn = nexradaws.NexradAwsInterface()


if __name__ == '__main__':
    start = dt(2022, 6, 2, 11, 0, 0)
    end = start + timedelta(hours=18)

    scans = conn.get_avail_scans_in_range(start, end, 'KHGX')
    scans = [scan for scan in scans if 'MDM' not in scan.filename and not scan.filename.endswith('j')]
    target = f'/Volumes/LtgSSD/nexrad_l2/{start.strftime("%Y%m%d")}'
    Path(target).mkdir(parents=True, exist_ok=True)
    conn.download(scans, target)