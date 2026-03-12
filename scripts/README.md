# Photo-to-Dataset Script

Convert a folder of letter images into a Zoogle Translate dataset.

## 1. Prepare your images

Each image should show **one letter/symbol** (A–Z). Use either layout:

### Option A: Folder per letter
```
dataset/
  A/
    001.png
    002.png
  B/
    001.png
  ...
```

### Option B: Flat naming
```
dataset/
  A_001.png
  A_002.png
  B_001.png
  ...
```

**Image format:**
- Black strokes on white background (default)
- 24×24 will be used internally; any size is fine (script resizes)
- PNG, JPG, BMP, WebP

## 2. Run the script

```bash
# Install Pillow if needed
pip install Pillow

# Folder layout (default)
python scripts/photo_to_dataset.py dataset/ -o my_language.json --name "My Alien Language"

# Flat layout
python scripts/photo_to_dataset.py dataset/ -o my_language.json --flat --name "My Alien Language"

# If your images have white strokes on black background
python scripts/photo_to_dataset.py dataset/ -o my_language.json --no-invert
```

## 3. Import in the app

1. Open Zoogle Translate
2. Go to **Setup & Train**
3. Click **Import Dataset**
4. Select the generated JSON file
