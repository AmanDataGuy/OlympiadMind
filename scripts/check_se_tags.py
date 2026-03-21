import xml.etree.ElementTree as ET
from pathlib import Path

path = Path(r"C:\Users\AMAND\Downloads\physics.stackexchange.com\Posts.xml")
count = 0
for event, elem in ET.iterparse(str(path), events=["end"]):
    if elem.tag == "row" and elem.get("PostTypeId") == "1":
        tags  = elem.get("Tags", "")
        score = int(elem.get("Score", 0))
        if score >= 5:
            print(f"Score {score}: {tags}")
            count += 1
        if count >= 10:
            break
    elem.clear()