"""
Fixed/Merged app.py
- Supabase-safe helpers for upload/download/get_public_url
- Extension fixes and filename sanitization
- In-memory CLIP feature extraction, merging, upload to Supabase
- Text/image search using features/features.npz from Supabase
- FIXED: CSV updates now properly overwrite old product data
"""
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import os
import io
import numpy as np
import pandas as pd
import re
from PIL import Image
from sqlalchemy import true
import torch
import clip
from supabase import create_client
from werkzeug.utils import secure_filename
from io import BytesIO
import math

# --------------------------
# Config
# --------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecretkey")

# Supabase credentials - use env vars in production
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://khrfhivjmaqopthxccxw.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImtocmZoaXZqbWFxb3B0aHhjY3h3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDQyMDUxMywiZXhwIjoyMDc1OTk2NTEzfQ.eGNCn5HZkhmHvES2Z6vtnLtxcs7a0_Mo6gNPEUpDjZc")
BUCKET = os.environ.get("SUPABASE_BUCKET", "uploads")

# Initialize Supabase client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    print("⚠️ Supabase initialization failed:", e)

# CLIP device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "Shoplly123")

# --------------------------
# Helpers: filenames / supabase operations
# --------------------------
def sanitize_filename(filename: str) -> str:
    name = secure_filename(filename)
    if not name:
        # fallback
        name = re.sub(r'[^\w\-_\.]', '_', filename)
    return name

def fix_extension(name: str) -> str:
    # normalize common miss-typed extensions
    name = re.sub(r'\.(jppg|jppg)$', '.jpg', name, flags=re.IGNORECASE)
    name = re.sub(r'\.(jjpg|jpe?g|jpeg)$', '.jpg', name, flags=re.IGNORECASE)
    # remove duplicate dots
    name = re.sub(r'\.{2,}', '.', name)
    return name

def safe_bytes_from_download(result):
    """
    Normalize different supabase client download returns into bytes or None.
    Some versions return raw bytes, some return a dict like {'data': b'...'},
    others return an object with .read()
    """
    if result is None:
        return None
    # bytes-like
    if isinstance(result, (bytes, bytearray)):
        return bytes(result)
    # dict with 'data'
    if isinstance(result, dict):
        if "data" in result and isinstance(result["data"], (bytes, bytearray)):
            return bytes(result["data"])
        # sometimes result might have {'error':...}
        return None
    # file-like object
    try:
        # e.g., requests.Response or similar
        if hasattr(result, "content") and isinstance(result.content, (bytes, bytearray)):
            return bytes(result.content)
        if hasattr(result, "read"):
            data = result.read()
            return bytes(data) if data is not None else None
    except Exception:
        pass
    return None

def upload_to_supabase_bytes(file_bytes: bytes, path: str, bucket: str = BUCKET, upsert: bool = True):
    """Upload raw bytes to Supabase storage. Returns public URL or None."""
    if supabase is None:
        print("⚠️ Supabase client not initialized.")
        return None
    try:
        path = path.lstrip("/")

        # Some versions support file_options={'upsert':True}, others need no 3rd arg
        try:
            res = supabase.storage.from_(bucket).upload(path, file_bytes, file_options={"upsert": upsert})
        except TypeError:
            # Fallback for older SDKs
            res = supabase.storage.from_(bucket).upload(path, file_bytes)

        return safe_get_public_url(path, bucket)
    except Exception as e:
        print(f"⚠️ Upload error for {path}: {e}")
        return None


def download_from_supabase_bytes(path: str, bucket: str = BUCKET):
    """Download file bytes from Supabase. Returns bytes or None."""
    if supabase is None:
        print("⚠️ Supabase client not initialized.")
        return None
    try:
        path = path.lstrip("/")
        result = supabase.storage.from_(bucket).download(path)
        data = safe_bytes_from_download(result)
        if data is None:
            # try the older interface: get_public_url then request (avoid external requests here) -> skip
            print(f"⚠️ Download returned no bytes for {path}.")
        return data
    except Exception as e:
        print(f"⚠️ Download error for {path}: {e}")
        return None

def safe_get_public_url(path: str, bucket: str = BUCKET):
    """Robustly return a public URL for a stored object.
       Handles supabase clients that return str or dict variants.
    """
    if supabase is None:
        return None
    try:
        path = path.lstrip("/")
        url = supabase.storage.from_(bucket).get_public_url(path)
        # If the SDK returns a string
        if isinstance(url, str):
            return url
        # If it returns a dict with variants of the key
        if isinstance(url, dict):
            for k in ("public_url", "publicURL", "publicUrl", "data"):
                if k in url and isinstance(url[k], str):
                    return url[k]
        # Last resort: try to build from SUPABASE_URL and bucket path (works for public buckets only)
        if SUPABASE_URL and bucket:
            # Supabase public URL format: <SUPABASE_URL>/storage/v1/object/public/<bucket>/<path>
            return f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{path}"
        return None
    except Exception as e:
        print(f"⚠️ get_public_url error for {path}: {e}")
        return None

def get_public_url(path: str, bucket: str = BUCKET):
    """Compatibility wrapper (keeps original name used in code)."""
    return safe_get_public_url(path, bucket)

# --------------------------
# CLIP feature helpers (in-memory, no local tmp)
# --------------------------
def load_features_from_supabase():
    """
    Download and load features/features.npz from Supabase.
    Returns (features ndarray, paths ndarray) or (None, None) if not found.
    """
    try:
        raw = download_from_supabase_bytes("features/features.npz")
        if not raw:
            print("⚠️ features/features.npz not found on Supabase.")
            return None, None
        with io.BytesIO(raw) as f:
            data = np.load(f, allow_pickle=True)
            feats = data.get("features") if "features" in data else data.get("arr_0")
            paths = data.get("paths") if "paths" in data else data.get("arr_1", None)
            if feats is None or paths is None:
                print("⚠️ features.npz format unexpected (missing keys).")
                return None, None
            return feats.astype(np.float32), paths.astype(object)
    except Exception as e:
        print("⚠️ Error loading features from Supabase:", e)
        return None, None

def save_features_to_supabase(features_array: np.ndarray, paths_array: np.ndarray):
    """
    Save provided features and paths as features/features.npz to Supabase.
    Overwrites existing file (upsert = True).
    """
    try:
        # Ensure arrays are proper dtype
        features_array = np.asarray(features_array, dtype=np.float32)
        paths_array = np.asarray(paths_array, dtype=object)

        # Deduplicate by path (keep first occurrence)
        try:
            unique_paths, idx = np.unique(paths_array, return_index=True)
            features_array = features_array[idx]
            paths_array = unique_paths
        except Exception:
            pass

        buf = io.BytesIO()
        # Save with explicit keys so load is predictable
        np.savez(buf, features=features_array, paths=paths_array)
        buf.seek(0)
        upload_to_supabase_bytes(buf.read(), "features/features.npz", upsert=True)
        print("✅ Uploaded features/features.npz to Supabase.")
        return True
    except Exception as e:
        print("⚠️ Failed to upload features to Supabase:", e)
        return False

def extract_clip_features_from_files(images_dict, batch_size=32):
    """
    images_dict: { folder_name: [FileStorage, FileStorage, ...], ... }
    Returns: (features ndarray, paths ndarray)
    Paths are strings like "folder/filename" or "filename"
    This function does not write any local files; works in-memory.
    """
    # load CLIP model
    os.environ["CLIP_CACHE_DIR"] = "/tmp/clip_cache"
    model, preprocess = clip.load("ViT-B/32", device=DEVICE, download_root="/tmp/clip_cache")
    model.eval()

    image_tensors = []
    image_paths = []

    for folder, files in images_dict.items():
        for f in files:
            try:
                # FileStorage: ensure we read bytes safely
                f.stream.seek(0)
                img_bytes = f.read()
                if not img_bytes:
                    print(f"⚠️ Empty file skipped: {getattr(f, 'filename', 'unknown')}")
                    continue
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = preprocess(img)  # single tensor
                image_tensors.append(tensor)
                fname = sanitize_filename(getattr(f, "filename", "") or "unnamed")
                fname = fix_extension(fname)
                relpath = f"{folder}/{fname}" if folder else fname
                relpath = relpath.lstrip("/")
                image_paths.append(relpath)
            except Exception as e:
                print(f"⚠️ Skipped image {getattr(f, 'filename', 'unknown')}: {e}")
                continue

    if not image_tensors:
        return None, None

    # batch encode
    all_feats = []
    for i in range(0, len(image_tensors), batch_size):
        batch = torch.stack(image_tensors[i:i+batch_size]).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.cpu().numpy().astype(np.float32))

    features_array = np.vstack(all_feats)
    paths_array = np.array(image_paths, dtype=object)
    return features_array, paths_array

# --------------------------
# Admin routes
# --------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        if request.form.get("password") == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("upload_page"))
        return render_template("admin_login.html", error="Incorrect password")
    return render_template("admin_login.html")

@app.route("/upload_page")
def upload_page():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))
    return render_template("index.html")

@app.route("/reload_products", methods=["POST"])
def reload_products():
    """Manual endpoint to reload product data from CSVs"""
    if not session.get("admin_logged_in"):
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        load_product_names()
        return jsonify({
            "success": True, 
            "message": f"Reloaded {len(product_name_map)} products"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --------------------------
# Upload route
# --------------------------
def extract_features_for_csv_filenames():
    """
    Read CSV filenames, find matching images (with full paths), extract CLIP features.
    Handles images in subfolders like images/dining/diningtable1.jpg
    """
    try:
        if not product_name_map:
            print("⚠️ No products loaded from CSV")
            return False
        
        print(f"📋 Found {len(product_name_map)} products in CSV")
        print("🔍 Scanning Supabase images folder for matching files...")
        
        csv_filenames = set(product_name_map.keys())
        
        # List all images in Supabase
        try:
            image_files = supabase.storage.from_("uploads").list("images", {"recursive": True})
        except Exception as e:
            print(f"⚠️ Could not list images folder: {e}")
            return False
        
        if not image_files:
            print("⚠️ No images found in Supabase")
            return False
        
        # Find matches - compare just the filename part with full paths stored
        matched_images = {}  # filename -> full_path
        for img_file in image_files:
            full_path = img_file.get("name", "")
            if not full_path or full_path.endswith("/"):
                continue
            
            filename = full_path.split("/")[-1]  # Get just the filename part
            
            if filename in csv_filenames:
                matched_images[filename] = full_path
                print(f"  ✅ Found: {filename} at {full_path}")
        
        if not matched_images:
            print("⚠️ No images matched CSV filenames")
            print(f"   CSV filenames: {list(csv_filenames)[:5]}...")  # Show sample
            return False
        
        print(f"📥 Downloading {len(matched_images)} matched images for feature extraction...")
        
        # Download and extract features
        image_list = []
        image_paths_list = []
        
        os.environ["CLIP_CACHE_DIR"] = "/tmp/clip_cache"
        model, preprocess = clip.load("ViT-B/32", device=DEVICE, download_root="/tmp/clip_cache")
        model.eval()
        
        for filename, full_path in matched_images.items():
            try:
                img_bytes = download_from_supabase_bytes(full_path)
                if not img_bytes:
                    print(f"  ⚠️ Could not download: {filename}")
                    continue
                
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = preprocess(img)
                image_list.append(tensor)
                image_paths_list.append(full_path)
                print(f"  ✅ Processed: {filename}")
                
            except Exception as e:
                print(f"  ⚠️ Error processing {filename}: {e}")
                continue
        
        if not image_list:
            print("⚠️ No images could be processed")
            return False
        
        print(f"🔄 Extracting CLIP features for {len(image_list)} images...")
        
        all_feats = []
        batch_size = 32
        
        for i in range(0, len(image_list), batch_size):
            batch = torch.stack(image_list[i:i+batch_size]).to(DEVICE)
            with torch.no_grad():
                feats = model.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.cpu().numpy().astype(np.float32))
        
        features_array = np.vstack(all_feats)
        paths_array = np.array(image_paths_list, dtype=object)
        
        print(f"💾 Saving {len(paths_array)} features to Supabase...")
        save_features_to_supabase(features_array, paths_array)
        
        print(f"✅ Features successfully updated: {len(paths_array)} images indexed")
        return True
        
    except Exception as e:
        print(f"⚠️ Error in extract_features_for_csv_filenames: {e}")
        return False


@app.route("/upload", methods=["POST"])
def upload_files():
    if not session.get("admin_logged_in"):
        return redirect(url_for("admin_login"))

    if not request.files:
        return "No files received", 400

    images_dict = {}
    csv_uploaded = False
    
    try:
        # CSV handling
        csv_file = request.files.get("csv_file")
        if csv_file:
            csv_file.stream.seek(0)
            csv_bytes = csv_file.read()
            csv_name = sanitize_filename(csv_file.filename or "metadata.csv")
            csv_name = fix_extension(csv_name)
            csv_remote = f"csv/{csv_name}"
            
            print(f"📤 Uploading CSV: {csv_remote}")
            upload_to_supabase_bytes(csv_bytes, csv_remote, upsert=True)
            csv_uploaded = True
            
            # Reload product names from new CSV
            print("🔄 Reloading product database...")
            load_product_names()
            print(f"✅ Product database updated: {len(product_name_map)} products in memory")
            
        # Handle image files
        for key in request.files:
            if key == "csv_file":
                continue
            file = request.files[key]
            folder_path, filename = os.path.split(key)
            fname = sanitize_filename(file.filename or filename or "unnamed")
            fname = fix_extension(fname)
            folder_name = folder_path or ""
            images_dict.setdefault(folder_name, []).append(file)

            try:
                file.stream.seek(0)
                bytes_data = file.read()
                pure_fname = fname.split("_", 1)[-1] if "_" in fname else fname
                remote_path = f"images/{folder_name}/{pure_fname}".replace("//", "/").lstrip("/")
                print(folder_name, fname, remote_path)
                upload_to_supabase_bytes(bytes_data, remote_path, upsert=True)
            except Exception as e:
                print(f"⚠️ Failed to upload image to Supabase: {e}")

        # Update features based on what was uploaded
        if csv_uploaded:
            print("\n" + "="*60)
            print("🔄 CSV UPLOADED - Rebuilding features.npz from CSV filenames")
            print("="*60)
            
            # Extract features for all files in CSV (whether images were uploaded or already exist)
            extract_features_for_csv_filenames()
        
        elif images_dict:
            # No CSV uploaded, but images were - keep old approach for backward compatibility
            print("🖼️ Processing new images for CLIP features...")
            new_feats, new_paths = extract_clip_features_from_files(images_dict)
            if new_feats is None or new_paths is None:
                return jsonify({"error": "No valid images to process for features"}), 400

            print(f"✅ Replacing features with {len(new_paths)} new images")
            save_features_to_supabase(new_feats, new_paths)

        return redirect(url_for("public_demo"))
        
    except Exception as e:
        print("⚠️ Upload Exception:", e)
        return jsonify({"error": str(e)}), 500
# --------------------------
# Public demo / homepage
# --------------------------
@app.route("/")
def public_demo():
    return render_template("result.html", session_id="static_session")


# --------------------------
# Text Search - CLIP text -> image similarity using features from Supabase
# --------------------------
import io
import pandas as pd

# Global dictionary to hold filename → {product_name, price}
product_name_map = {}

def load_product_names():
    """
    Load and merge ALL CSV files from Supabase.
    ✅ Processes oldest → newest, so newer CSVs override older ones
    ✅ New attributes from newer CSVs are added
    ✅ Old filenames kept if not in newer CSV
    """
    global product_name_map
    try:
        files = supabase.storage.from_("uploads").list("csv")

        if not files or len(files) == 0:
            print("⚠️ No CSV files found in Supabase.")
            product_name_map = {}
            return

        # Sort by updated_at - OLDEST first, so newer ones overwrite
        files_sorted = sorted(files, key=lambda f: f.get("updated_at", ""))

        merged_data = {}
        files_processed = 0
        products_loaded = 0

        # Process each CSV from oldest to newest
        for f in files_sorted:
            filename = f.get("name", "")
            if not filename.lower().endswith(".csv"):
                continue

            csv_path = f"csv/{filename}"
            updated_at = f.get("updated_at", "unknown")
            
            print(f"📦 Loading CSV: {csv_path} (updated: {updated_at})")

            try:
                response = supabase.storage.from_("uploads").download(csv_path)
                if not response:
                    print(f"⚠️ Could not download {filename}")
                    continue

                csv_data = io.StringIO(response.decode("utf-8"))
                df = pd.read_csv(
                    csv_data, 
                    on_bad_lines='skip',
                    engine='python',
                    keep_default_na=False,
                    na_values=[''],
                    dtype=str
                )
                
                df = df.replace({np.nan: '', 'nan': '', 'NaN': '', 'None': '', 'none': ''})

                files_processed += 1
                rows_from_this_csv = 0

                for _, row in df.iterrows():
                    file_key = row.get("filename", "").strip()
                    if not file_key:
                        continue

                    # Create record from this CSV
                    record = {}
                    for col in df.columns:
                        val = row.get(col, "")
                        if pd.isna(val):
                            val_str = ""
                        else:
                            val_str = str(val).strip()
                        record[col] = val_str
                    
                    record.setdefault("product_name", file_key)
                    record.setdefault("price", "—")

                    # If filename exists, merge records smartly
                    if file_key in merged_data:
                        old_record = merged_data[file_key]
                        merged_record = old_record.copy()
                        
                        # Update with new values and add new columns
                        for col, val in record.items():
                            if val:  # non-empty value from new CSV
                                merged_record[col] = val
                            elif col not in merged_record:
                                merged_record[col] = val
                        
                        merged_data[file_key] = merged_record
                        print(f"  🔄 Updated: {file_key}")
                    else:
                        merged_data[file_key] = record
                        print(f"  ✨ New: {file_key}")
                    
                    rows_from_this_csv += 1
                    products_loaded += 1

                print(f"   ✅ {rows_from_this_csv} products from this CSV")

            except Exception as inner_e:
                print(f"⚠️ Error loading {filename}:", inner_e)

        product_name_map = merged_data
        print(f"\n✅ Loaded {products_loaded} total products from {files_processed} CSV files")
        print(f"   (Newer CSVs override older ones for same filename)")

    except Exception as e:
        print("⚠️ Error loading product names:", e)
        product_name_map = {}
# Load product names once on startup
load_product_names()

import numpy as np
import pandas as pd
import math

# ==========================================
# Helper Functions
# ==========================================

def normalize_confidence(similarity_score):
    """
    Convert CLIP similarity to human-friendly confidence percentage.
    Uses exponential scaling for better discrimination.
    """
    # Typical CLIP similarity range: 0.23-0.35
    low, high = 0.23, 0.33
    normalized = np.clip((similarity_score - low) / (high - low), 0, 1)
    # Apply exponential curve (power < 1 spreads out values)
    confidence = (normalized ** 0.6) * 100
    return round(confidence, 2)

def normalize_image_search_confidence(similarity_score):
    """
    Convert CLIP similarity to human-friendly confidence percentage specifically for image search.
    Uses a more conservative approach to avoid inflated confidence scores.
    """
    # For image search, use a more conservative range
    # CLIP image-to-image similarity can be higher than text-to-image
    low, high = 0.25, 0.45
    
    # Ensure we don't go below 0
    if similarity_score < low:
        return 0.0
    
    # Normalize to 0-1 range
    normalized = min((similarity_score - low) / (high - low), 1.0)
    
    # Apply a more conservative exponential curve
    confidence = (normalized ** 0.8) * 100
    
    # Cap at 95% to avoid perfect scores
    return round(min(confidence, 95.0), 2)


def parse_price_query_no_regex(query: str):
    """Parse price filters from natural language"""
    query_lower = query.lower()
    min_price, max_price = None, None
    words = query_lower.split()

    if "under" in words or "below" in words or "less" in words:
        for w in words:
            if w.isdigit():
                max_price = int(w)
                break
    elif "above" in words or "over" in words or "more" in words or "greater" in words:
        for w in words:
            if w.isdigit():
                min_price = int(w)
                break
    elif "between" in words:
        try:
            idx = words.index("between")
            nums = [int(w) for w in words[idx+1:idx+4] if w.isdigit()]
            if len(nums) >= 2:
                min_price, max_price = nums[0], nums[1]
        except Exception:
            pass
    elif "cheap" in words or "affordable" in words or "budget" in words:
        max_price = 500
    elif "expensive" in words or "premium" in words or "luxury" in words:
        min_price = 1000

    # Remove price-related words
    filtered_words = [
        w for w in words if not w.isdigit() and w not in
        ["under", "below", "less", "above", "over", "more", "greater", 
         "between", "and", "to", "cheap", "expensive", "affordable", 
         "premium", "luxury", "budget"]
    ]
    text_query = " ".join(filtered_words).strip()
    
    return text_query, min_price, max_price


# ==========================================
# TEXT SEARCH ROUTE (Original CSV Logic)
# ==========================================

def get_product_info(stored_path, paths):
    """
    Robustly get product info from CSV using the stored path.
    Handles various filename formats and naming inconsistencies.
    """
    filename = str(stored_path).split("/")[-1]  # e.g., "diningtable1.jpg"
    
    # Try exact match first
    if filename in product_name_map:
        return product_name_map[filename]
    
    # Try without underscores (in case path had prefix)
    clean_filename = filename.split("_", 1)[-1]
    if clean_filename in product_name_map:
        return product_name_map[clean_filename]
    
    # Try case-insensitive match
    filename_lower = filename.lower()
    for key, val in product_name_map.items():
        if key.lower() == filename_lower:
            return val
    
    # Try partial match - look for any key that ends with this filename
    for key, val in product_name_map.items():
        if key.endswith(filename) or filename.endswith(key):
            return val
    
    # Return empty dict if no match found
    print(f"⚠️ No CSV match found for: {filename} (checked against {len(product_name_map)} products)")
    return {}

# ==========================================
# UPDATED TEXT SEARCH ROUTE
# ==========================================

@app.route("/chat_search", methods=["POST"])
def chat_search():
    try:
        load_product_names()
        
        payload = request.get_json() or {}
        query = payload.get("query", "").strip()
        if not query:
            return jsonify({"error": "Query required"}), 400

        parsed_text, min_price, max_price = parse_price_query_no_regex(query)
        if parsed_text:
            query = parsed_text
        if not query:
            return jsonify({"error": "Query required"}), 400

        keywords = query.split()

        feats, paths = load_features_from_supabase()
        if feats is None or paths is None:
            return jsonify({"error": "No features found"}), 404

        os.environ["CLIP_CACHE_DIR"] = "/tmp/clip_cache"
        model, _ = clip.load("ViT-B/32", device=DEVICE, download_root="/tmp/clip_cache")
        model.eval()

        # CSV matching per keyword
        keyword_matches = []
        for kw in keywords:
            kw_lower = kw.lower()
            matches = set()
            for idx, stored_path in enumerate(paths):
                product_info = get_product_info(stored_path, paths)
                # Check if keyword exists in ANY CSV field
                for val in product_info.values():
                    if isinstance(val, str) and kw_lower in val.lower():
                        matches.add(idx)
                        break
            keyword_matches.append(matches)

        matched_keywords = [kw for i, kw in enumerate(keywords) if keyword_matches[i]]
        unmatched_keywords = [kw for i, kw in enumerate(keywords) if not keyword_matches[i]]
        csv_matched_sets = [s for s in keyword_matches if s]
        
        sims = np.zeros(len(paths))
        
        if not unmatched_keywords:
            print(f"✅ All keywords found in CSV: {matched_keywords}")
            final_indices = set.intersection(*csv_matched_sets) if csv_matched_sets else set()
            
        elif unmatched_keywords and csv_matched_sets:
            print(f"✅ CSV keywords: {matched_keywords}, CLIP keywords: {unmatched_keywords}")
            
            clip_query = " ".join(unmatched_keywords)
            text_tokens = clip.tokenize([clip_query]).to(DEVICE)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy().astype(np.float32)

            sims = feats @ text_features.T
            sims = sims.squeeze()
            
            threshold = 0.50
            valid_idx = np.where(sims >= threshold)[0]
            clip_matched_indices = set(valid_idx)
            
            csv_union = set.union(*csv_matched_sets)
            final_indices = csv_union & clip_matched_indices
            
        else:
            print(f"⚠️ No CSV matches, passing all to CLIP: {keywords}")
            final_indices = set()

        if not final_indices:
            print(f"⚠️ No results, falling back to full CLIP search")
            text_tokens = clip.tokenize([query]).to(DEVICE)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.cpu().numpy().astype(np.float32)

            sims = feats @ text_features.T
            sims = sims.squeeze()
            
            threshold = 0.27
            valid_idx = np.where(sims >= threshold)[0]
            final_indices = set(valid_idx)

        if not final_indices:
            return jsonify({"results": [{
                "filename": "no_results.png",
                "product_name": "No matching items found",
                "price": "—",
                "confidence": "0%",
                "url": "#"
            }]})

        results = []

        for idx in sorted(final_indices, key=lambda i: sims[i] if i < len(sims) else 0, reverse=True)[:10]:
            stored_path = str(paths[idx])
            product_info = get_product_info(stored_path, paths)
            
            product_name = product_info.get("product_name", str(stored_path).split("/")[-1])
            price = product_info.get("price", "—")
            price_display = f"${price}" if price not in [None, "—", ""] else "—"

            raw_sim = float(sims[idx]) if idx < len(sims) else 0.0
            
            is_csv_match = False
            if csv_matched_sets:
                csv_union = set.union(*csv_matched_sets) if csv_matched_sets else set()
                is_csv_match = idx in csv_union
            
            if raw_sim == 0.0 and is_csv_match:
                confidence = 95.0
            elif raw_sim > 0.0:
                confidence = normalize_confidence(raw_sim)
                if is_csv_match:
                    confidence = min(confidence * 1.1, 99.5)
            else:
                confidence = 50.0

            price_val = None
            try:
                if isinstance(price, (int, float)):
                    price_val = float(price)
                elif isinstance(price, str) and price.strip() not in ["", "—", "-"]:
                    pclean = "".join([c for c in price if c.isdigit() or c == "."])
                    if pclean:
                        price_val = float(pclean)
            except Exception:
                pass

            if min_price is not None and (price_val is None or price_val < min_price):
                continue
            if max_price is not None and (price_val is None or price_val > max_price):
                continue

            filename = str(stored_path).split("/")[-1]
            folder_name = filename.split("_", 1)[0] if "_" in filename else ""
            remote_image_path = f"images/{folder_name}/{filename.split('_', 1)[-1]}".replace("//", "/").lstrip("/")
            public_url = get_public_url(remote_image_path) or "#"

            result_item = {
                "filename": filename,
                "product_name": product_name,
                "price": price_display,
                "confidence": f"{confidence}%",
                "url": public_url
            }

            for key, value in product_info.items():
                if key.lower() in ["filename", "url"]:
                    continue
                if value is None:
                    continue
                
                try:
                    if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
                        continue
                except Exception:
                    pass
                
                if isinstance(value, str):
                    v = value.strip()
                    if not v or v.lower() in ['nan', 'none', 'null', '—', '-']:
                        continue
                    result_item[key] = v
                else:
                    result_item[key] = value

            results.append(result_item)

        return jsonify({"results": results})

    except Exception as e:
        print("⚠️ Search Exception:", e)
        return jsonify({"error": str(e)}), 500


# ==========================================
# UPDATED IMAGE SEARCH ROUTE
# ==========================================

@app.route("/image_search", methods=["POST"])
def image_search():
    try:
        load_product_names()
        
        if "query_image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["query_image"]
        file.stream.seek(0)
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "Empty image uploaded"}), 400
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        feats, paths = load_features_from_supabase()
        if feats is None or paths is None:
            return jsonify({"error": "No features found"}), 404

        os.environ["CLIP_CACHE_DIR"] = "/tmp/clip_cache"
        model, preprocess = clip.load("ViT-B/32", device=DEVICE, download_root="/tmp/clip_cache")
        model.eval()

        with torch.no_grad():
            processed = preprocess(img).unsqueeze(0).to(DEVICE)
            query_feat = model.encode_image(processed)
            query_feat /= query_feat.norm(dim=-1, keepdim=True)
            query_feat = query_feat.cpu().numpy().astype(np.float32)

        sims = feats @ query_feat.T
        sims = sims.squeeze()

        threshold = 0.30
        valid_indices = np.where(sims >= threshold)[0]
        
        if valid_indices.size == 0:
            return jsonify({"results": [{
                "filename": "no_results.png",
                "product_name": "No matching items found",
                "price": "—",
                "confidence": "0%",
                "url": "#"
            }]})

        top_idx = sims.argsort()[::-1][:10]

        results = []
        for i in top_idx:
            if i >= len(paths) or sims[i] < threshold:
                continue

            stored_path = str(paths[i])
            product_info = get_product_info(stored_path, paths)
            
            product_name = product_info.get("product_name", str(stored_path).split("/")[-1])
            price = product_info.get("price", "—")
            price_display = f"${price}" if price not in [None, "—", ""] else "—"

            raw_similarity = float(sims[i])
            confidence = normalize_image_search_confidence(raw_similarity)
            
            print(f"🔍 Image search: {product_name} | Raw similarity: {raw_similarity:.4f} | Confidence: {confidence}%")

            filename = str(stored_path).split("/")[-1]
            folder_name = filename.split("_", 1)[0] if "_" in filename else ""
            remote_image_path = f"images/{folder_name}/{filename.split('_', 1)[-1]}".replace("//", "/").lstrip("/")
            public_url = get_public_url(remote_image_path) or "#"

            result_item = {
                "filename": filename,
                "product_name": product_name,
                "price": price_display,
                "confidence": f"{confidence}%",
                "url": public_url
            }

            for key, value in product_info.items():
                if key.lower() in ["filename", "url"]:
                    continue
                if value is None:
                    continue
                
                try:
                    if pd.isna(value) or (isinstance(value, float) and math.isnan(value)):
                        continue
                except Exception:
                    pass
                
                if isinstance(value, str):
                    v = value.strip()
                    if not v or v.lower() in ['nan', 'none', 'null', '—', '-']:
                        continue
                    result_item[key] = v
                else:
                    result_item[key] = value

            results.append(result_item)

        if not results:
            return jsonify({"results": [{
                "filename": "no_results.png",
                "product_name": "No matching items found",
                "price": "—",
                "confidence": "0%",
                "url": "#"
            }]})

        return jsonify({"results": results})

    except Exception as e:
        print("⚠️ image_search Exception:", e)
        return jsonify({"error": str(e)}), 500
# --------------------------
# Serve uploads compatibility route
# --------------------------
@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    remote = f"images/{filename}".lstrip("/")
    public_url = get_public_url(remote)
    if public_url:
        return redirect(public_url)
    return jsonify({"error": "File not found"}), 404

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    try:
        feats, paths = load_features_from_supabase()
        if feats is not None and paths is not None:
            print(f"✅ Found {len(paths)} features on Supabase at startup.")
    except Exception:
        pass

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))