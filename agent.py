def publishing_king(state: AgentState):
    print("👑 Publisher: Finalizing and Saving...")
    timestamp = int(time.time())
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    success = False
    # 1. Image Logic (simplified for stability)
    try:
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=f"Futuristic tech visual: {state['topic']}",
            config=types.GenerateImagesConfig(number_of_images=1)
        )
        if response.generated_images:
            with open(img_path, "wb") as f:
                f.write(response.generated_images[0].image_bytes)
            success = True
    except:
        print("   ⚠️ Nano Banana skipped.")

    if not success:
        try:
            encoded = urllib.parse.quote(state['topic'][:50])
            res = requests.get(f"https://image.pollinations.ai/prompt/{encoded}", timeout=15)
            if res.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(res.content)
                success = True
        except: pass

    # 2. File Saving Logic (STABLE VERSION)
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    date_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
    # Clean the slug for the filename
    safe_slug = re.sub(r'[^a-z0-9-]', '', state['topic'].lower().replace(' ', '-'))[:50]
    
    post_filename = f"{date_str}-{safe_slug}.md"
    post_path = os.path.join("_posts", post_filename)
    os.makedirs("_posts", exist_ok=True)

    # Building the content safely without complex f-string nesting
    header = "---\n"
    header += f"title: \"{state['topic']}\"\n"
    header += f"date: {date_full}\n"
    header += f"categories: [{state['field']}]\n"
    header += f"tags: [{state['seo_keywords']}]\n"
    header += f"image:\n  path: /assets/img/{img_filename}\n"
    header += "---\n\n"
    
    final_markdown = header + state['full_draft']
    
    with open(post_path, "w", encoding='utf-8') as f:
        f.write(final_markdown)
        
    print(f"   ✅ ARTICLE SAVED: {post_path}")
    return {"content": final_markdown, "image_url": img_filename}
