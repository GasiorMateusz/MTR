import xml.etree.ElementTree as ET

from lib.array import Array


# Funkcja do analizy pliku GPX i wyodrębnienia współrzędnych x, y, z
def import_gpx(plik):
    coor_x = []
    coor_y = []
    coor_z = []

    tree = ET.parse(plik)
    root = tree.getroot()

    for trkpt in root.findall(".//{http://www.topografix.com/GPX/1/1}trkpt"):
        x = float(trkpt.get("lon"))
        y = float(trkpt.get("lat"))

        ele = trkpt.find("{http://www.topografix.com/GPX/1/1}ele")
        z = float(ele.text) if ele is not None else None
        k = 3
        k_1 = 0.1 * k
        k_2 = 100 * k
        coor_x.append(x*k_2)
        coor_y.append(y*k_2)
        coor_z.append(z*k_1)
    cutoff = int(len(coor_x)/1)
    list = [coor_x[:cutoff],coor_y[:cutoff],coor_z[:cutoff]]
    set = Array(list).transpose()
    return set, list



if __name__ == "__main__":
    x, y, z = import_gpx("car_trip.gpx")
    print("Współrzędne X:", x)
    print("Współrzędne Y:", y)
    print("Współrzędne Z:", z)