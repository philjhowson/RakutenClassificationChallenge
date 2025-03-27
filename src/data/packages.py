import argostranslate.package
import argostranslate.translate
from argostranslate import package

# Update package index to fetch available language models
print("🔄 Updating package index...")
package.update_package_index()

# List of supported languages for translation to English
supported_languages = ["fr", "de", "es", "it", "sv", "pl", "nl", "ro", "pt", "ja"]

# Find and install relevant packages
available_packages = package.get_available_packages()
for pkg in available_packages:
    from_code = pkg.from_code
    to_code = pkg.to_code

    if from_code in supported_languages and to_code == "en":
        print(f"🔽 Installing language pack: {from_code} → {to_code} ...")
        package.install_from_path(pkg.download())
        print(f"✅ Installed: {from_code} → {to_code}")

print("🎉 All requested language packs installed successfully!")
