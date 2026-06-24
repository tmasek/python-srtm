from pathlib import Path
from typing import List, Tuple, Callable
from zipfile import ZipFile
from math import floor

from srtm.utilities import get_srtm3_file_path, get_srtm1_file_path
from srtm.base_coordinates import RasterBaseCoordinates


# From: <https://stackoverflow.com/a/8662355/2505304>
def bilinear_interpolation(x: float, y: float, points: List[Tuple[float, float, float]]) -> float:
    """
    Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    """
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)


class HeightMap:
    """Provides access to a single SRTM HGT file

    Data will be lazy-loaded on first access
    """

    raster: bytes = None
    base_coordinates: RasterBaseCoordinates
    file_path_fn: Callable = None
    expected_values = 1442401
    values_per_row = 1201

    def __init__(self, path: Path, base_coordinates: RasterBaseCoordinates = None):
        self.path = path
        self.base_coordinates = (
            base_coordinates or RasterBaseCoordinates.from_file_path(path)
        )

        # We subtract one as each row overlaps the neighbouring raster by 1 pixel
        self.pixel_width = 1 / (self.values_per_row - 1)

    @classmethod
    def from_base_coordinates(cls, base_coordinates: RasterBaseCoordinates):
        return cls(
            path=cls.file_path_fn(base_coordinates.file_name),
            base_coordinates=base_coordinates,
        )

    def ensure_loaded(self, force=False):
        """Ensure the file has been loaded from disk"""
        if not force and self.raster is not None:
            return

        if ".zip" in self.path.suffixes:
            zipped_files = ZipFile(self.path).namelist()
            zipped_files = [name for name in zipped_files if ".hgt" in name]
            assert len(zipped_files) == 1, (
                f"ZIP at {self.path} contains the wrong number of hgt files "
                f"({len(zipped_files)}!=1). Contains {zipped_files}"
            )
            self.raster = ZipFile(self.path).read(zipped_files[0])
        else:
            self.raster = self.path.read_bytes()

        self.validate()

    def validate(self):
        """Perform sanity checks"""
        expected_bytes = self.expected_values * 2
        assert len(self.raster) == expected_bytes, (
            f"Unexpected number of bytes found in {self.path}. "
            f"Expected {expected_bytes}, found {len(self.raster)}"
        )

    def get_altitude_for_pixel(self, x: int, y: int) -> int:
        """Get the height at the given pixel

        Will trigger loading of data
        """
        self.ensure_loaded()
        # Get the 1-indexed pixel number
        pixel_number = x + (y - 1) * self.values_per_row
        # Convert it to be 0-indexed
        pixel_number -= 1

        byte_number = pixel_number * 2
        return int.from_bytes(
            self.raster[byte_number : byte_number + 2], byteorder="big"
        )

    # def get_altitude_for_latitude_and_longitude(
    #     self, latitude: float, longitude: float
    # ) -> int:
    #     """Get the height at the given lat/lng"""
    #     x, y = self._latitude_and_longitude_to_coordinates(latitude, longitude)
    #     return self.get_altitude_for_pixel(x, y)

    _D_STEP = 0.0

    def _latitude_and_longitude_to_coordinates(
        self, latitude: float, longitude: float
    ) -> Tuple[int, int]:
        """Convert the given lat/long into x/y coordinates for this SRTM data"""
        origin_latitude = self.base_coordinates.latitude + 1
        origin_longitude = self.base_coordinates.longitude
        latitude_offset = origin_latitude - latitude + self._D_STEP*self.pixel_width
        longitude_offset = longitude - origin_longitude + self._D_STEP*self.pixel_width

        if latitude_offset > 1 or latitude_offset < 0:
            raise ValueError(
                f"Latitude {latitude} with offset {latitude_offset} is not within "
                f"this heightmap of base coordinates {self.base_coordinates}"
            )
        if longitude_offset > 1 or longitude_offset < 0:
            raise ValueError(
                f"Longitude {longitude} with offset {longitude_offset} is not within "
                f"this heightmap of base coordinates {self.base_coordinates}"
            )

        # Add one because pixels are 1-indexed
        x = floor(longitude_offset / self.pixel_width) + 1
        y = floor(latitude_offset / self.pixel_width) + 1

        return x, y

    def _coordinates_to_latitude_and_longitude(
            self, x: int, y: int
    ) -> Tuple[float, float]:
        """Convert the given x/y coordinates into lat/long for this SRTM data"""
        origin_latitude = self.base_coordinates.latitude + 1
        origin_longitude = self.base_coordinates.longitude

        longitude_offset = (x - 1)*self.pixel_width
        latitude_offset = (y - 1)*self.pixel_width

        latitude = origin_latitude - latitude_offset + self._D_STEP*self.pixel_width
        longitude = origin_longitude + longitude_offset - self._D_STEP*self.pixel_width
        
        return latitude, longitude

    def get_altitude_for_latitude_and_longitude(
        self, latitude: float, longitude: float
    ) -> int:
        """Get the height at the given lat/lng"""
        x, y = self._latitude_and_longitude_to_coordinates(latitude, longitude)
        lat1, lon0 = self._coordinates_to_latitude_and_longitude(x, y)
        lat0, lon1 = self._coordinates_to_latitude_and_longitude(x+1, y+1)
        assert lat0 <= latitude <= lat1
        assert lon0 <= longitude <= lon1
        return bilinear_interpolation(latitude, longitude,
                                      [(lat0, lon0, self.get_altitude_for_pixel(x,   y+1)),
                                       (lat1, lon0, self.get_altitude_for_pixel(x,   y  )),
                                       (lat0, lon1, self.get_altitude_for_pixel(x+1, y+1)),
                                       (lat1, lon1, self.get_altitude_for_pixel(x+1, y  ))])


class Srtm1HeightMap(HeightMap):
    """Provides access to a single SRTM HGT file

    Data will be lazy-loaded on first access
    """

    raster: bytes = None
    base_coordinates: RasterBaseCoordinates

    expected_values = 12967201
    values_per_row = 3601
    file_path_fn = get_srtm1_file_path


class Srtm3HeightMap(HeightMap):
    """Provides access to a single SRTM HGT file

    Data will be lazy-loaded on first access
    """

    raster: bytes = None
    base_coordinates: RasterBaseCoordinates
    expected_values = 1442401
    values_per_row = 1201
    file_path_fn = get_srtm3_file_path
