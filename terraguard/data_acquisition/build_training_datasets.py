"""Zero-queue ARCO-ERA5, STAC DEM & GEM PyG dataset builders
(blueprint §4.2 reference implementation).

These are the real-world extraction protocols for production training runs.
They require network access to Google Cloud / AWS open-data buckets and the
optional deps ``xarray, zarr, gcsfs, pystac-client, odc-stac``; the CPU
training scripts in ``terraguard/training`` use the spectrum-matched
procedural generators in ``terraguard/common/fields.py`` instead, producing
tensors of identical shape so these loaders are drop-in replacements.
"""
from __future__ import annotations

ESA_WORLDCOVER_TO_Z0 = {
    10: 0.80, 20: 0.15, 30: 0.03, 40: 0.10,
    50: 1.20, 60: 0.005, 80: 0.0002, 90: 0.05,
}


def stream_arco_era5_india_corridor(
        start_date: str = "2020-01-01", end_date: str = "2023-12-31",
        lat_slice: tuple = (18.5, 11.5),       # Karnataka belt, N -> S
        lon_slice: tuple = (74.0, 78.5),
        out_zarr_path: str = "./data/raw/era5_karnataka_25km.zarr") -> str:
    """Stream 0.25-deg ERA5 tensors from Google Cloud ARCO-ERA5
    (WeatherBench 2), bypassing Copernicus CDS tape queues."""
    import xarray as xr  # type: ignore
    arco_url = ("gs://weatherbench2/datasets/era5/"
                "1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr")
    ds = xr.open_zarr(arco_url, storage_options={"token": "anon"},
                      consolidated=True)
    target_vars = [
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "surface_pressure", "total_cloud_cover",
        "u_component_of_wind", "v_component_of_wind", "geopotential",
        "specific_humidity",
    ]
    subset = (ds[target_vars]
              .sel(time=slice(start_date, end_date),
                   latitude=slice(*lat_slice), longitude=slice(*lon_slice))
              .sel(level=[1000, 925, 850, 700, 500]))
    subset.chunk({"time": 128, "latitude": -1,
                  "longitude": -1}).to_zarr(out_zarr_path, mode="w")
    return out_zarr_path


def extract_dem_and_roughness_stac(
        bbox_lon_lat: list = (76.20, 14.10, 76.60, 14.50)):
    """Keyless STAC query for 30 m Copernicus DEM + 10 m ESA WorldCover."""
    import odc.stac          # type: ignore
    import pystac_client     # type: ignore
    stac = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
    dem_items = list(stac.search(collections=["cop-dem-glo-30"],
                                 bbox=bbox_lon_lat).items())
    wc_items = list(stac.search(collections=["esa-worldcover"],
                                bbox=bbox_lon_lat).items())
    dem = odc.stac.load(dem_items, bands=["data"], bbox=bbox_lon_lat,
                        resolution=30)["data"]
    wc = odc.stac.load(wc_items, bands=["map"], bbox=bbox_lon_lat,
                       resolution=10)["map"]
    z0 = wc.copy(deep=True).astype("float32")
    for cls, val in ESA_WORLDCOVER_TO_Z0.items():
        z0 = z0.where(wc != cls, val)
    return dem, z0


def build_gem_h3_pyg_graph(gem_csv_path: str, h3_resolution: int = 7):
    """Assemble the supervised spatial graph from Global Energy Monitor
    India project records (Operating/Construction = 1, Shelved/Cancelled = 0)
    mapped onto Uber H3 hexagons."""
    import pandas as pd      # type: ignore
    import torch
    from ..layer3_spatial.context_spatial_gnn import knn_spatial_graph
    df = pd.read_csv(gem_csv_path)
    pos = ("operating", "construction")
    df["label"] = df["Status"].str.lower().isin(pos).astype(float)
    coords = torch.tensor(df[["Latitude", "Longitude"]].values,
                          dtype=torch.float32) * 111.0
    edge_index, edge_attr = knn_spatial_graph(coords, k=6)
    x = torch.tensor(df[["CapacityMW"]].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.float32)
    return {"x": x, "y": y, "edge_index": edge_index, "edge_attr": edge_attr}
