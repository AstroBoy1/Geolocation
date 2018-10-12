import s2sphere
import folium
import math


def main():
    r = s2sphere.RegionCoverer()
    p1 = s2sphere.LatLng.from_degrees(48.831776, 2.222639)
    p2 = s2sphere.LatLng.from_degrees(48.902839, 2.406)
    cell_ids = r.get_covering(s2sphere.LatLngRect.from_point_pair(p1, p2))
    # print(cell_ids)

    # create a map
    map_osm = folium.Map(location=[48.86, 2.3], zoom_start=12, tiles='Stamen Toner')

    # # get vertices from rect to draw them on map
    # rect_vertices = []
    # for i in [0, 1, 2, 3, 0]:
    #     rect_vertices.append([vertex.lat().degrees(), vertex.lng().degrees()])

    # draw the cells
    style_function = lambda x: {'weight': 1, 'fillColor': '#eea500'}
    for cellid in cell_ids:
        cell = s2sphere.Cell(cellid)
        vertices = []
        for i in range(0, 4):
            vertex = cell.get_vertex(i)
            latlng = s2sphere.LatLng.from_point(vertex)
            # currently the angle is in radians
            vertices.append((math.degrees(latlng.lat().radians), math.degrees(latlng.lng().radians)))
            gj = folium.GeoJson({"type": "Polygon", "coordinates": [vertices]}, style_function=style_function)
            gj.add_child(folium.Popup(cellid.to_token()))
            gj.add_to(map_osm)

            # # warning PolyLine is lat,lng based while GeoJSON is not
            # ls = folium.PolyLine(rect_vertices, color='red', weight=2)
            # ls.add_children(folium.Popup("shape"))
            # ls.add_to(map_osm)


if __name__ == "__main__":
    main()
