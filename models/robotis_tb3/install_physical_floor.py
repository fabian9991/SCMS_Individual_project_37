"""Build a sibling scene from the current assignment XML; all paths stay relative."""
from pathlib import Path
import xml.etree.ElementTree as ET


def install():
    folder = Path(__file__).resolve().parent
    source = folder / 'scene_assignment.xml'
    target = folder / 'scene_assignment_plate.xml'
    tree = ET.parse(source, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    root = tree.getroot()
    body = root.find('worldbody')
    floor = body.find("geom[@name='floor']") if body is not None else None
    if floor is None or floor.get('type') != 'plane' or floor.get('material') != 'groundplane':
        raise ValueError('Expected the original assignment plane named floor with material groundplane.')
    body.remove(floor)
    root.append(ET.Element('include', file='physical_floor.xml'))
    tree.write(target, encoding='utf-8', xml_declaration=True)
    print('Created', target)
    return target


if __name__ == '__main__':
    install()
