import os
import time
import inspect
import requests
import math

from ._common import print_page_error

   # Upload media >10 images
   # Change pagination to 10
   # Search & create a saved search section
   # Optional- Go to media, bookmark it? or last visited?
def test_features(request, page_factory, project):
   print("Project Detail Page Feature tests...")
   page = page_factory(
       f"{os.path.basename(__file__)}__{inspect.stack()[0][3]}")
   page.goto(f"/{project}/project-detail")
   page.on("pageerror", print_page_error)

   page.select_option('.pagination select.form-select', value="100")
   # page.wait_for_selector('text="Page 1 of 1"')
   time.sleep(5)

   # Initial card length
   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   initialCardLength = len(cards)
   newCardsLength = 15
   totalCards = initialCardLength + newCardsLength

   nasa_space_photo_1 = '/tmp/hubble-sees-the-wings-of-a-butterfly.jpg'
   if not os.path.exists(nasa_space_photo_1):
      url = 'https://images-assets.nasa.gov/image/hubble-sees-the-wings-of-a-butterfly-the-twin-jet-nebula_20283986193_o/hubble-sees-the-wings-of-a-butterfly-the-twin-jet-nebula_20283986193_o~small.jpg'
      with requests.get(url, stream=True) as r:
         r.raise_for_status()
         with open(nasa_space_photo_1, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
               if chunk:
                  f.write(chunk)

   nasa_space_photo_2 = '/tmp/layers-in-galle-crater.jpg'
   if not os.path.exists(nasa_space_photo_2):
      url = 'https://images-assets.nasa.gov/image/PIA21575/PIA21575~medium.jpg'
      with requests.get(url, stream=True) as r:
         r.raise_for_status()
         with open(nasa_space_photo_2, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
               if chunk:
                  f.write(chunk)

   nasa_space_photo_3 = '/tmp/behemoth-black-hole.jpg'
   if not os.path.exists(nasa_space_photo_3):
      url = 'https://images-assets.nasa.gov/image/behemoth-black-hole-found-in-an-unlikely-place_26209716511_o/behemoth-black-hole-found-in-an-unlikely-place_26209716511_o~medium.jpg'
      with requests.get(url, stream=True) as r:
         r.raise_for_status()
         with open(nasa_space_photo_3, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
               if chunk:
                  f.write(chunk)

   page.set_input_files('section-upload input', [nasa_space_photo_1,nasa_space_photo_2,nasa_space_photo_3,nasa_space_photo_2,nasa_space_photo_2,nasa_space_photo_3,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1,nasa_space_photo_1])
   page.query_selector('upload-dialog').query_selector('text=Close').click()

   page.click('reload-button')
   page.wait_for_selector('media-card')
   time.sleep(5)

   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards) # existing + new cards

   print(f"Length of cards {cardLength}  == should match totalCards {totalCards}")
   assert cardLength == totalCards

   # Test selecting less cards
   page.select_option('.pagination select.form-select', value="10")
   pages = int(math.ceil(totalCards / 10))
   page.wait_for_selector(f'text="Page 1 of {str(pages)}"')
   time.sleep(5)
   
   cardsHidden = page.query_selector_all('media-card[style="display: none; visibility: visible;"]')
   cardsHiddenLength = len(cardsHidden)

   print(f"Length of cards hidden {cardsHiddenLength}  == totalCards - 10 {totalCards - 10}")
   totalMinus = totalCards - 10
   assert cardsHiddenLength == totalMinus

   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards)

   print(f"Visible card length {cardLength}  == 10")
   assert cardLength == 10
   
   # Test pagination
   paginationLinks = page.query_selector_all('.pagination a')
   paginationLinks[2].click()
   page.wait_for_selector(f'text="Page 2 of {pages}"')
   time.sleep(5)
   
   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards)
   totalOnSecond = totalCards - 10
   if totalOnSecond > 10:
      totalOnSecond = 10
   print(f"Second page length of cards {cardLength}  == {totalOnSecond}")
   assert cardLength == totalOnSecond


   href = cards[0].query_selector('a').get_attribute('href')

   # Click off the page to test the url history
   if 'annotation' in href:
      print(f"Clicking the first card to annotator....")
      cards[0].query_selector('a').click()
      page.wait_for_selector('.annotation__panel h3')
      page.go_back()

      page.wait_for_selector('media-card')
      print(f"Is pagination preserved?")

      cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
      cardLength = len(cards)
      totalOnSecond = totalCards - 10
      if totalOnSecond > 10:
         totalOnSecond = 10
      print(f"(refreshed) Second page length of cards {cardLength}  == {totalOnSecond}")
      assert cardLength == totalOnSecond

   # Test filtering
   page.click('text="Filter"')
   page.wait_for_selector('filter-condition-group button.btn.btn-outline.btn-small')
   page.click('filter-condition-group button.btn.btn-outline.btn-small')

   page.wait_for_selector('enum-input[name="Field"]')
   page.select_option('enum-input[name="Field"] select', value="filename")

   page.wait_for_selector('text-input[name="Value"] input')
   page.fill('text-input[name="Value"] input', "black\-hole")

   filterGroupButtons = page.query_selector_all('.modal__footer button')
   filterGroupButtons[0].click()

   page.wait_for_selector('text="Page 1 of 1"')
   time.sleep(5)

   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards)
   print(f"Cards length after search {cardLength} == 2")
   assert cardLength == 2

   saveSearch = page.query_selector('text="Add current search"')
   saveSearch.click()

   newSectionFromSearch = "Black Holes"
   page.wait_for_selector('.modal__main input[placeholder="Give it a name..."]')
   page.fill('.modal__main input[placeholder="Give it a name..."]', newSectionFromSearch)
   saveButton = page.query_selector('text="Save"')
   saveButton.click()


   page.wait_for_selector(f'text="{newSectionFromSearch}"')
   print(f'New section created named: {newSectionFromSearch}')

   clearSearch = page.query_selector('removable-pill button')
   clearSearch.click()

   page.wait_for_selector(f'text="{totalCards} Files"')
   time.sleep(5)

   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards)
   print(f"After search cleared cardLength {cardLength} == 10")
   assert cardLength == 10

   page.query_selector(f'text="{newSectionFromSearch}"').click()
   page.wait_for_selector('text="2 Files"')
   time.sleep(5)
   
   cards = page.query_selector_all('media-card[style="display: block; visibility: visible;"]')
   cardLength = len(cards)
   print(f"Cards in saved section {cardLength} == 2")
   assert cardLength == 2