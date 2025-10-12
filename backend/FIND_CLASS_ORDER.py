"""
Quick guide to help you find the correct class label order for your model.
This explains how to check your training setup.
"""

print("="*80)
print("🔍 HOW TO FIND YOUR CORRECT CLASS LABEL ORDER")
print("="*80)

print("""
Your model predictions are wrong because the CLASS_LABELS dictionary in app.py
doesn't match the order your model was trained with.

📁 METHOD 1: Check Your Training Folder Structure
─────────────────────────────────────────────────────────────────────────────

If you used ImageDataGenerator with flow_from_directory(), the class order is
determined by alphabetical folder names:

Example:
    Training/
    ├── glioma/          ← Class 0 (alphabetically first)
    ├── meningioma/      ← Class 1
    ├── no_tumor/        ← Class 2
    └── pituitary/       ← Class 3

The class indices are assigned alphabetically!

💡 ACTION: Check your training folder and list subfolders alphabetically.


📝 METHOD 2: Check Your Training Code
─────────────────────────────────────────────────────────────────────────────

Look for lines like this in your training notebook/script:

    train_generator = train_datagen.flow_from_directory(
        'Training',
        target_size=(299, 299),
        batch_size=32,
        class_mode='categorical'
    )
    
    print(train_generator.class_indices)  ← THIS SHOWS THE MAPPING!

Output example:
    {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}


🧪 METHOD 3: Test With Known Images
─────────────────────────────────────────────────────────────────────────────

1. Get 4 MRI images, one for each tumor type (glioma, meningioma, pituitary, no_tumor)
2. Run: python test_xception_config.py <path_to_glioma_image.jpg>
3. Note which class index (0, 1, 2, or 3) it predicts
4. Repeat for other tumor types
5. This reveals the true mapping!


📊 COMMON CONFIGURATIONS
─────────────────────────────────────────────────────────────────────────────

Config A (Alphabetical - Most Common):
    0: Glioma
    1: Meningioma
    2: No Tumor         ← "no_tumor" comes before "pituitary" alphabetically
    3: Pituitary

Config B (Alternative):
    0: Glioma
    1: Meningioma
    2: Pituitary
    3: No Tumor

Config C (Binary - if only 2 classes):
    0: No Tumor
    1: Tumor


🔧 HOW TO FIX app.py
─────────────────────────────────────────────────────────────────────────────

Once you know the correct order:

1. Open app.py
2. Find the CLASS_LABELS dictionary (around line 23-33)
3. Update it to match your training order

Example: If your training shows {'glioma': 0, 'meningioma': 1, 'no_tumor': 2, 'pituitary': 3}

    CLASS_LABELS = {
        0: "Glioma Tumor",        # Must be at index 0
        1: "Meningioma Tumor",    # Must be at index 1
        2: "No Tumor",            # Must be at index 2  ← NOTE THE ORDER!
        3: "Pituitary Tumor"      # Must be at index 3
    }

4. Save and restart Flask server


⚡ QUICK TEST COMMAND
─────────────────────────────────────────────────────────────────────────────

Run this to test both preprocessing methods and see predictions:

    python test_xception_config.py path/to/known_tumor_image.jpg

This will show you:
    ✅ Which preprocessing method works ([0,1] vs [-1,1])
    ✅ Which class index the model predicts
    ✅ Confidence scores for all classes


🎯 FINAL CHECKLIST
─────────────────────────────────────────────────────────────────────────────

□ Found correct class order from training folder or code
□ Updated CLASS_LABELS in app.py to match
□ Tested with test_xception_config.py to verify
□ Updated preprocessing method if needed ([0,1] vs [-1,1])
□ Restarted Flask server (python3 app.py)
□ Tested with real MRI images
□ Verified Grad-CAM highlights correct regions


If still wrong after this, the model itself may be poorly trained! 🚨
""")

print("="*80)
print("\n💡 Need help? Run: python test_xception_config.py <test_image.jpg>")
print("="*80 + "\n")
