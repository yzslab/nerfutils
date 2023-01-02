import os
import glob
import PIL.Image
import PIL.ExifTags
import xml.etree.ElementTree as ET


def extract_exif_values(im):
    raw_exif_data = im._getexif()
    friendly_exif_data = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in im._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    return friendly_exif_data, raw_exif_data


# https://exiftool.org/TagNames/DJI.html
def extract_xmp_values(im):
    xmp = {}

    for segment, content in im.applist:
        # fine xmp content
        marker, body = content.split(b'\x00', 1)
        if segment == 'APP1' and marker == b'http://ns.adobe.com/xap/1.0/':
            # convert to string
            str_body = body.decode()
            tree = ET.fromstring(str_body)
            for child in tree.iter():
                for attr in child.attrib:
                    key = attr.rsplit("}", maxsplit=1)
                    if len(key) == 2:
                        key = key[1]
                    else:
                        key = attr
                    xmp[key] = child.attrib[attr]
            # process row by row
            # for row in str_body.split("\n"):
            #     # retrieve xmp items
            #     stripped_row = row.strip(" ")
            #     if stripped_row.startswith("drone-dji"):
            #         _, item = stripped_row.split(":")
            #         k, v = item.split("=")
            #         v = v.strip("\"")
            #         # store into xml dictionary
            #         xmp[k] = v

    return xmp


def gps_tuple_to_angle(i) -> float:
    return (((float(i[2]) / 60.) + float(i[1])) / 60.) + float(i[0])


def extract(image_path: str):
    with PIL.Image.open(image_path) as im:
        exif, _ = extract_exif_values(im)

        gpsinfo = {}
        for key in exif['GPSInfo'].keys():
            decode = PIL.ExifTags.GPSTAGS.get(key, key)
            gpsinfo[decode] = exif['GPSInfo'][key]
        exif["GPSInfo"] = gpsinfo

        exif["SimplifiedGPSInfo"] = {
            "LatitudeRef": gpsinfo["GPSLatitudeRef"],
            "LongitudeRef": gpsinfo["GPSLongitudeRef"],
            "AltitudeRef": int.from_bytes(gpsinfo["GPSAltitudeRef"], "little"),
            "Latitude": gps_tuple_to_angle(gpsinfo["GPSLatitude"]),
            "Longitude": gps_tuple_to_angle(gpsinfo["GPSLongitude"]),
            "Altitude": float(gpsinfo["GPSAltitude"]),
        }

        xmp = extract_xmp_values(im)
        exif["xmp"] = xmp

    return exif


def extract_by_directory(path: str):
    img_exif = {}

    # get image filename in directory
    image_list = glob.glob(os.path.join(path, "*.jpg"))
    image_list += glob.glob(os.path.join(path, "*.JPG"))
    for image_path in image_list:
        img_exif[image_path] = extract(image_path)

    return img_exif


def parse_exif_values_by_directory(path: str):
    img_exif = extract_by_directory(path)
    for k in img_exif:
        exif = img_exif[k]
        xmp = exif["xmp"]
        if len(xmp.keys()) == 0:
            continue
        # exif["xmp"] = {
        #     "gps": {
        #         "latitude": float(xmp["GpsLatitude"]),
        #         "longitude": float(xmp["GpsLongitude"]),
        #         "absolute_altitude": float(xmp["AbsoluteAltitude"]),
        #         "relative_altitude": float(xmp["RelativeAltitude"]),
        #     },
        #     "gimbal": {
        #         "roll": float(xmp["GimbalRollDegree"]),
        #         "yaw": float(xmp["GimbalYawDegree"]),
        #         "pitch": float(xmp["GimbalPitchDegree"]),
        #     },
        #     "flight": {
        #         "roll": float(xmp["FlightRollDegree"]),
        #         "yaw": float(xmp["FlightYawDegree"]),
        #         "pitch": float(xmp["FlightPitchDegree"]),
        #     },
        # }

    return img_exif
