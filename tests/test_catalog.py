import fitsio
import lensPT as lpt


def test_catalog():
    print("testing for catalog initialization")
    data = fitsio.read("./fpfs-cut32-0000-g1-0000.fits")
    colnames = [
        "fpfs_M00",
        "fpfs_M20",
        "fpfs_M22c",
        "fpfs_M22s",
        "fpfs_M40",
        "fpfs_M42c",
        "fpfs_M42s",
    ]
    lpt.observable.Catalog(data, mode_names=colnames)
    lpt.observable.Catalog(data)
    lpt.observable.Catalog(
            "./fpfs-cut32-0000-g1-0000.fits",
            mode_names=colnames
            )
    cat = lpt.observable.Catalog("./fpfs-cut32-0000-g1-0000.fits")
    cat.mode_names
    cat.data
    return

if __name__ == "__main__":
    test_catalog()
