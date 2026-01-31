# frozen_string_literal: true

source "https://rubygems.org"

# Jekyll 4.3.x is the stable standard for GitHub Pages
gem "jekyll", "~> 4.3.4"
gem "jekyll-theme-chirpy", "~> 7.0"

# This gem is REQUIRED to make 'remote_theme' work
gem "jekyll-remote-theme"

group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
  gem "jekyll-archives"
  gem "jekyll-redirect-from"
  gem "jekyll-twitter-plugin"
end

# Support for newer Ruby versions
gem "csv"
gem "base64"

# Keep html-proofer in the test group for CI/CD checks
group :test do
  gem "html-proofer", "~> 5.0"
end

# Platform specific gems for local compatibility (Windows/JRuby)
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
