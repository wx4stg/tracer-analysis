import nexradaws
from glob import glob
import sys
from datetime import datetime as dt
from os import path
from pathlib import Path
from datetime import datetime as dt, timedelta
conn = nexradaws.NexradAwsInterface()


if __name__ == '__main__':
    start = dt.strptime(sys.argv[1], '%Y-%m-%dT%H:%M')
    end = start + timedelta(hours=int(sys.argv[2]))
    radar_sites = sys.argv[3:]
    for site in radar_sites:
        scans = conn.get_avail_scans_in_range(start, end, site)
        scans = [scan for scan in scans if 'MDM' not in scan.filename and not scan.filename.endswith('j')]
        target = f'/Volumes/LtgSSD/nexrad_l2/{start.strftime("%Y%m%d")}'
        already_downloaded = glob(f'{target}/*')
        already_downloaded = [path.basename(adl) for adl in already_downloaded]
        scans = [scan for scan in scans if scan.filename not in already_downloaded]
        Path(target).mkdir(parents=True, exist_ok=True)
        conn.download(scans, target)
