# frozen_string_literal: true

source "https://rubygems.org"

# Jekyll 4.3.x is the stable standard for GitHub Pages
gem "jekyll", "~> 4.3.4"

# This gem is REQUIRED to make 'remote_theme' work in your _config.yml
gem "jekyll-remote-theme"

group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
  
  # --- Suggested Additions ---
  # Generates archive pages for tags and categories (common for Chirpy)
  gem "jekyll-archives"
  # Allows for redirecting old URLs to new ones
  gem "jekyll-redirect-from"
  # Renders tweets using a Liquid tag (if needed for your posts)
  gem "jekyll-twitter-plugin"
end

# Keep html-proofer in the test group for CI/CD checks
group :test do
  gem "html-proofer", "~> 5.0"
end

# Platform specific gems for local compatibility (Windows/JRuby)
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Specific fix for Windows file watching
gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]

# If you are using Ruby 3.4+, you may need these as they were removed from the stdlib:
# gem "csv"
# gem "base64"
