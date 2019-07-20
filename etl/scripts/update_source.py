# -*- coding: utf-8 -*-

import concurrent.futures
import threading
import os.path as osp
import pandas as pd
from ddf_utils.factory.common import requests_retry_session

out_dir = '../source/'
metadata_file = '../source/metadata.xlsx'
download_url_prefix = 'https://population.un.org/wpp/Download/Files/1_Indicators (Standard)/EXCEL_FILES/'


fileNameAndURLMapping = {
    'POP': '1_Population',
    'FERT': '2_Fertility',
    'MORT': '3_MortaLity',
    'MIGR': '4_Migrantion',
    'INT': '5_Interpolated'
}

thread_local = threading.local()

def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests_retry_session()
    return thread_local.session


def get_url(fname: str):
    for k, v in fileNameAndURLMapping.items():
        if fname.startswith(f'WPP2019_{k}'):
            return download_url_prefix + v + '/' + fname
    print('URL for {} is undefned'.format(fname))
    return None


def download(url):
    session = get_session()
    fname = url.split('/')[-1]
    if osp.exists(osp.join('../source/', fname)):
        return
    res = session.get(url, stream=True)
    res.raise_for_status()
    with open(osp.join(out_dir, fname), 'wb') as f:
        for chunk in res.iter_content(chunk_size=512 * 1024):
            if chunk:
                f.write(chunk)
    f.close()
    print('downloaded: {}'.format(fname))


def main():
    meta = pd.read_excel(metadata_file, sheet_name='New')
    files = meta['file'].tolist()
    urls = [get_url(x) for x in files]

    # also add location metadata
    urls.append('https://population.un.org/wpp2019/Download/Files/4_Metadata/WPP2019_F01_LOCATIONS.XLSX')

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(download, urls)


if __name__ == '__main__':
    main()
