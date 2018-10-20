import urllib
from google.appengine.api import images
import cloudstorage as gcs
import PIL

image_at_url = urllib.urlopen(url)
content_type =  image_at_url.headers['Content-Type']
filename = #use your own or get from file

image_bytes = image_at_url.read()
image_at_url.close()
image = images.Image(image_bytes)

# this comes in handy if you want to resize images:
# if image.width > 800 or image.height > 800:
#     image_bytes = images.resize(image_bytes, 800, 800)

options={'x-goog-acl': 'public-read', 'Cache-Control': 'private, max-age=0, no-transform'}
gcs.
with gcs.open(filename, 'w', content_type=content_type, options=options) as f:
    f.write(image_bytes)
    f.close()
