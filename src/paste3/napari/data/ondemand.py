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
        "url": "https://dl.dropboxusercontent.com/scl/fi/zq0dlcgjaxfe9fqbp0hf4/patient_2_slice_0.h5ad?rlkey=sxj5c843b38vd3iv2n74824hu&st=pdelsbuz&dl=1",
        "hash": "md5:3f2a599a067d3752bd735ea2a01e19f3",
    },
    "paste3_sample_patient_2_slice_1.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/a5ufhjylxfnvcn5sw4yp0/patient_2_slice_1.h5ad?rlkey=p6dp78qhz6qrh0ut49s7b3fvj&st=2ysuoay4&dl=1",
        "hash": "md5:a6d1db8ae803e52154cb47e7f8433ffa",
    },
    "paste3_sample_patient_2_slice_2.h5ad": {
        "url": "https://dl.dropboxusercontent.com/scl/fi/u7aaq9az8sia26cn4ac4s/patient_2_slice_2.h5ad?rlkey=3ynobd5ajhlvc7lwdbyg0akj1&st=fp7aq5zh&dl=1",
        "hash": "md5:7a64c48af327554dd314439fdbe718ce",
    },
}


def get_file(which):
    assert which in files, f"Unknown file {which}"
    file = files[which]
    return pooch.retrieve(
        url=file["url"],
        known_hash=file["hash"],
        fname=which,
        processor=file.get("processor"),
        path=CACHE_PATH,
    )
