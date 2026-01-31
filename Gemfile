# frozen_string_literal: true

source "https://rubygems.org"

# Chirpy v7.x handles Jekyll 4.3.x dependencies automatically
gem "jekyll-theme-chirpy", "~> 7.0"

group :jekyll_plugins do
  # These are usually the only extra ones you might need 
  # if not already bundled or if you want specific versions.
  gem "jekyll-archives"
  gem "jekyll-paginate"
  gem "jekyll-redirect-from"
  gem "jekyll-twitter-plugin"
end

# Support for newer Ruby versions (3.x+)
gem "csv"
gem "base64"

group :test do
  gem "html-proofer", "~> 5.0"
end

# Platform specific gems for local compatibility
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]
