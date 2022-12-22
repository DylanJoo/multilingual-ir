def chunk_list(datas, chunksize):
    for i in range(0, len(datas), chunksize):
        yield datas[i:i + chunksize]
