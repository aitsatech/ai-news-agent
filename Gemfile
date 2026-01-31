source "https://rubygems.org"

# Jekyll 4.3.x is the stable standard for GitHub Pages
gem "jekyll", "~> 4.3.4"

# Use the theme as a gem to ensure the build environment has the assets
gem "jekyll-theme-chirpy", "~> 7.1.1"

group :jekyll_plugins do
  gem "jekyll-feed"
  gem "jekyll-seo-tag"
  gem "jekyll-sitemap"
  gem "jekyll-paginate"
  gem "jekyll-include-cache"
end

# Keep html-proofer in the test group as you have it in your workflow
group :test do
  gem "html-proofer", "~> 5.0"
end

# Platform specific gems for local compatibility
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]
