import pooch

"""
Large files that are downloaded/cached automagically using the pooch library.

  How to add new entries to this list:

  Get expected hash by running md5sum or md5 on the file
  For Dropbox links, use "Copy Link" to get the URL,
    replace "www.dropbox.com" with "dl.dropboxusercontent.com",
    replace "dl=0" with "dl=1"
"""

CACHE_PATH = pooch.os_cache("paste3")

files = {
    "paste3_sample_patient_2_slice_0.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/zq0dlcgjaxfe9fqbp0hf4/paste3_sample_patient_2_slice_0.h5ad?rlkey=sxj5c843b38vd3iv2n74824hu&st=wcy6oxbt&dl=1",
        "hash": "md5:3f2a599a067d3752bd735ea2a01e19f3",
    },
    "paste3_sample_patient_2_slice_1.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/a5ufhjylxfnvcn5sw4yp0/paste3_sample_patient_2_slice_1.h5ad?rlkey=p6dp78qhz6qrh0ut49s7b3fvj&st=tyamjq8b&dl=1",
        "hash": "md5:a6d1db8ae803e52154cb47e7f8433ffa",
    },
    "paste3_sample_patient_2_slice_2.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/u7aaq9az8sia26cn4ac4s/paste3_sample_patient_2_slice_2.h5ad?rlkey=3ynobd5ajhlvc7lwdbyg0akj1&st=0l2nw8i2&dl=1",
        "hash": "md5:7a64c48af327554dd314439fdbe718ce",
    },
    "paste3_sample_patient_5_slice_0.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/ypj05gsopwh74ruycjll8/paste3_sample_patient_5_slice_0.h5ad?rlkey=fdbdpuncunpcmqyxed5x687t6&st=u61mutsr&dl=1",
        "hash": "md5:d74b47b1e8e9af45085a76c463169f75",
    },
    "paste3_sample_patient_5_slice_1.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/e1cqlwi313ykjgzl8pmoi/paste3_sample_patient_5_slice_1.h5ad?rlkey=g60hoh2d6qpleaqb59m4xr8n4&st=whwe4oxr&dl=1",
        "hash": "md5:cfa621ccb3d13181bd82edc58de4ba22",
    },
    "paste3_sample_patient_5_slice_2.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/0jim40rezs1kfk0hhx8r7/paste3_sample_patient_5_slice_2.h5ad?rlkey=gu5wh4m2i58so35gwiyvpvkmj&st=qhpnoqjg&dl=1",
        "hash": "md5:2c93234cf9592a6a3d79afe08f59a144",
    },
    "paste3_sample_patient_9_slice_0.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/auu0hj6b7b7adhek23eal/paste3_sample_patient_9_slice_0.h5ad?rlkey=k2cwouul7pk7zjeymn6ryt2ef&st=sdseia87&dl=1",
        "hash": "md5:6854eed7b4dc768007ca91e4c7ea35df",
    },
    "paste3_sample_patient_9_slice_1.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/4nt6rd70u7czsftv0ka0u/paste3_sample_patient_9_slice_1.h5ad?rlkey=b79bmtp9dz48u9oa4tlkq3uqg&st=bvlw62ym&dl=1",
        "hash": "md5:a8734cf971d08b851a4502c96f7b56a5",
    },
    "paste3_sample_patient_9_slice_2.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/o9jhgkwzzppgfsuewy6eo/paste3_sample_patient_9_slice_2.h5ad?rlkey=kjl6jn24awtwafhoz9yjamdob&st=r7le0v8j&dl=1",
        "hash": "md5:87df8ce67e0b3ce0891a868d565e2216",
    },
    "paste3_sample_patient_10_slice_0.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/b7s1hfkfy3ajtzb0wy1yb/paste3_sample_patient_10_slice_0.h5ad?rlkey=u77grnq5xud7q7wwzm2hq05hx&st=icmc2ttg&dl=1",
        "hash": "md5:3dacdb5d8b39056d1b764b401b94cbac",
    },
    "paste3_sample_patient_10_slice_1.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/4nlwrpllzows1u0rodi7j/paste3_sample_patient_10_slice_1.h5ad?rlkey=6m3qzp0gqwrgpahlm3qn1o2w5&st=gondhkyg&dl=1",
        "hash": "md5:ec8abf1a6a0cf6e8f5841b5dad15bdb7",
    },
    "paste3_sample_patient_10_slice_2.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/qjb82v0kqirkr00x0acyf/paste3_sample_patient_10_slice_2.h5ad?rlkey=z8j0pb57le9h6h802dcn9t9i2&st=qxt6a0bw&dl=1",
        "hash": "md5:1b60fbd0bf267babb0f0c0434bdc5d21",
    },
}


def get_file(which: str) -> list[str]:
    remote_files = {k: v for k, v in files.items() if k.startswith(which)}
    return [
        pooch.retrieve(
            url=v["url"],
            known_hash=v["hash"],
            fname=k,
            processor=v.get("processor"),
            path=CACHE_PATH,
        )
        for k, v in remote_files.items()
    ]
