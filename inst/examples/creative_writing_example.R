# Creative Writing and Content Generation with edgemodelr
# Explore the creative capabilities of local LLM models

library(edgemodelr)

cat("✨ Creative Writing with edgemodelr\n")
cat(rep("=", 60), "\n\n")

# Setup creative writing model
setup_creative_writer <- function() {
  cat("Setting up creative writing model...\n")
  
  tryCatch({
    if (!interactive()) {
      cat("Demo mode: Using placeholder creative content\n")
      return(NULL)
    }
    
    # Use a model good for creative tasks
    setup <- edge_quick_setup("llama3.2-1b")
    
    if (!is.null(setup$context)) {
      cat("✅ Creative writer ready!\n\n")
      return(setup$context)
    } else {
      cat("❌ Failed to load model\n")
      return(NULL)
    }
    
  }, error = function(e) {
    cat("❌ Setup failed:", e$message, "\n")
    return(NULL)
  })
}

# Story Generation
generate_story <- function(ctx, genre, theme, length = "short") {
  if (is.null(ctx)) return("Demo: A creative story would be generated here based on the genre and theme.")
  
  n_predict <- switch(length,
    "short" = 200,
    "medium" = 400, 
    "long" = 600,
    200
  )
  
  prompt <- paste0(
    "Write a ", length, " ", genre, " story about ", theme, ". ",
    "Make it engaging with vivid descriptions and interesting characters.\n\n",
    "Story:"
  )
  
  story <- edge_completion(
    ctx,
    prompt,
    n_predict = n_predict,
    temperature = 0.8,  # High creativity
    top_p = 0.9
  )
  
  result <- sub(".*Story:", "", story)
  return(trimws(result))
}

# Poetry Generation
generate_poem <- function(ctx, style, topic, form = "free verse") {
  if (is.null(ctx)) return("Demo: A beautiful poem would be created here.")
  
  prompt <- paste0(
    "Write a ", form, " poem in ", style, " style about ", topic, ". ",
    "Focus on imagery, emotion, and literary devices.\n\n",
    "Poem:"
  )
  
  poem <- edge_completion(
    ctx,
    prompt,
    n_predict = 150,
    temperature = 0.9,  # Very creative
    top_p = 0.8
  )
  
  result <- sub(".*Poem:", "", poem)
  return(trimws(result))
}

# Character Development
create_character <- function(ctx, character_type, setting) {
  if (is.null(ctx)) return("Demo: A detailed character profile would be created.")
  
  prompt <- paste0(
    "Create a detailed character profile for a ", character_type, 
    " in a ", setting, " setting. Include:\n",
    "- Name and appearance\n",
    "- Personality traits\n", 
    "- Background story\n",
    "- Motivations and goals\n",
    "- Unique quirks or abilities\n\n",
    "Character Profile:"
  )
  
  character <- edge_completion(
    ctx,
    prompt,
    n_predict = 300,
    temperature = 0.8,
    top_p = 0.9
  )
  
  result <- sub(".*Character Profile:", "", character)
  return(trimws(result))
}

# Dialogue Generation
generate_dialogue <- function(ctx, character1, character2, situation) {
  if (is.null(ctx)) return("Demo: A dramatic dialogue would be written here.")
  
  prompt <- paste0(
    "Write a dialogue between ", character1, " and ", character2, 
    " in this situation: ", situation, ". ",
    "Make the dialogue natural, revealing character personalities and advancing the plot.\n\n",
    "Dialogue:"
  )
  
  dialogue <- edge_completion(
    ctx,
    prompt,
    n_predict = 250,
    temperature = 0.8,
    top_p = 0.9
  )
  
  result <- sub(".*Dialogue:", "", dialogue)
  return(trimws(result))
}

# World Building
build_world <- function(ctx, genre, key_elements) {
  if (is.null(ctx)) return("Demo: A rich fictional world would be described here.")
  
  elements_str <- paste(key_elements, collapse = ", ")
  
  prompt <- paste0(
    "Create a detailed world for a ", genre, " story. Include these elements: ",
    elements_str, ". Describe:\n",
    "- Geography and environment\n",
    "- Culture and society\n",
    "- Technology or magic system\n",
    "- History and conflicts\n",
    "- Unique aspects\n\n",
    "World Description:"
  )
  
  world <- edge_completion(
    ctx,
    prompt,
    n_predict = 400,
    temperature = 0.8,
    top_p = 0.9
  )
  
  result <- sub(".*World Description:", "", world)
  return(trimws(result))
}

# Plot Development
develop_plot <- function(ctx, genre, main_character, conflict) {
  if (is.null(ctx)) return("Demo: A compelling plot outline would be developed.")
  
  prompt <- paste0(
    "Develop a ", genre, " plot outline featuring ", main_character, 
    " facing this conflict: ", conflict, ". Include:\n",
    "- Setup and introduction\n",
    "- Rising action and complications\n", 
    "- Climax\n",
    "- Resolution\n",
    "- Character development arc\n\n",
    "Plot Outline:"
  )
  
  plot <- edge_completion(
    ctx,
    prompt,
    n_predict = 350,
    temperature = 0.7,
    top_p = 0.9
  )
  
  result <- sub(".*Plot Outline:", "", plot)
  return(trimws(result))
}

# Creative Writing Prompts Generator
generate_writing_prompts <- function(ctx, num_prompts = 5, theme = "general") {
  if (is.null(ctx)) {
    return(c(
      "Demo prompt 1: A time traveler gets stuck in the wrong era",
      "Demo prompt 2: The last library in the world",
      "Demo prompt 3: Colors start disappearing from the world"
    ))
  }
  
  prompt <- paste0(
    "Generate ", num_prompts, " creative writing prompts about ", theme, ". ",
    "Make them intriguing and specific enough to inspire stories. ",
    "Format as a numbered list.\n\n",
    "Writing Prompts:"
  )
  
  prompts <- edge_completion(
    ctx,
    prompt,
    n_predict = 200,
    temperature = 0.8,
    top_p = 0.9
  )
  
  result <- sub(".*Writing Prompts:", "", prompts)
  return(trimws(result))
}

# Interactive Story Builder
interactive_story_builder <- function() {
  cat("📚 Interactive Story Builder\n")
  cat("Build a story step by step with AI assistance!\n\n")
  
  ctx <- setup_creative_writer()
  if (is.null(ctx)) {
    cat("Story builder unavailable in demo mode\n")
    return(NULL)
  }
  
  story_elements <- list()
  
  # Get story parameters from user
  cat("Let's build your story together!\n\n")
  
  # In interactive R session, these would prompt user input
  # For demo purposes, using default values
  story_elements$genre <- if (interactive()) readline("📖 What genre? (fantasy, sci-fi, mystery, etc.): ") else "fantasy"
  story_elements$setting <- if (interactive()) readline("🌍 Where does it take place?: ") else "enchanted forest"
  story_elements$protagonist <- if (interactive()) readline("🦸 Who is the main character?: ") else "young wizard"
  story_elements$conflict <- if (interactive()) readline("⚔️ What's the main conflict?: ") else "dark magic threatens the realm"
  
  cat("\n🔨 Building your story elements...\n\n")
  
  # Generate story components
  cat("Creating main character profile...\n")
  character_profile <- create_character(ctx, story_elements$protagonist, story_elements$setting)
  story_elements$character_profile <- character_profile
  cat("✅ Character created!\n\n")
  
  cat("Developing plot outline...\n")
  plot_outline <- develop_plot(ctx, story_elements$genre, story_elements$protagonist, story_elements$conflict)
  story_elements$plot <- plot_outline
  cat("✅ Plot developed!\n\n")
  
  cat("Building the world...\n")
  world_desc <- build_world(ctx, story_elements$genre, c(story_elements$setting, "unique cultures"))
  story_elements$world <- world_desc
  cat("✅ World built!\n\n")
  
  # Generate opening scene
  cat("Writing opening scene...\n")
  opening_prompt <- paste0(
    "Write the opening scene of a ", story_elements$genre, " story set in ", story_elements$setting,
    " featuring ", story_elements$protagonist, ". Use vivid descriptions and hook the reader."
  )
  
  opening_scene <- edge_completion(
    ctx,
    opening_prompt,
    n_predict = 300,
    temperature = 0.8,
    top_p = 0.9
  )
  story_elements$opening <- opening_scene
  
  # Present the complete story framework
  cat("\n", rep("=", 60), "\n")
  cat("🎉 YOUR STORY FRAMEWORK IS COMPLETE!\n") 
  cat(rep("=", 60), "\n\n")
  
  cat("📖 GENRE:", story_elements$genre, "\n")
  cat("🌍 SETTING:", story_elements$setting, "\n") 
  cat("🦸 PROTAGONIST:", story_elements$protagonist, "\n")
  cat("⚔️ CONFLICT:", story_elements$conflict, "\n\n")
  
  cat("👤 CHARACTER PROFILE:\n", character_profile, "\n\n")
  cat("📋 PLOT OUTLINE:\n", plot_outline, "\n\n")
  cat("🌐 WORLD DESCRIPTION:\n", world_desc, "\n\n")
  cat("📝 OPENING SCENE:\n", opening_scene, "\n\n")
  
  edge_free_model(ctx)
  return(story_elements)
}

# Creative Writing Workshop
creative_workshop <- function() {
  cat("🎨 Creative Writing Workshop\n")
  cat(rep("-", 40), "\n\n")
  
  ctx <- setup_creative_writer()
  
  # Workshop exercises
  exercises <- list(
    list(
      title = "Character Voice Exercise",
      description = "Write the same scene from three different character perspectives",
      scene = "discovering a mysterious letter"
    ),
    list(
      title = "Sensory Description Challenge", 
      description = "Describe a place using all five senses",
      scene = "an abandoned carnival at midnight"
    ),
    list(
      title = "Dialogue Mastery",
      description = "Write a conversation that reveals character without exposition",
      scene = "two strangers stuck in an elevator"
    ),
    list(
      title = "Genre Blending",
      description = "Combine two different genres in one scene", 
      scene = "romance meets horror"
    )
  )
  
  for (i in seq_along(exercises)) {
    exercise <- exercises[[i]]
    
    cat("\n", rep("-", 30), "\n")
    cat("Exercise", i, ":", exercise$title, "\n")
    cat(rep("-", 30), "\n")
    cat("Description:", exercise$description, "\n")
    cat("Scene:", exercise$scene, "\n\n")
    
    if (!is.null(ctx)) {
      prompt <- paste0(
        "Creative writing exercise: ", exercise$description, ". ",
        "Scene: ", exercise$scene, ". ",
        "Write a compelling example that demonstrates this technique.\n\n",
        "Example:"
      )
      
      example <- edge_completion(
        ctx,
        prompt,
        n_predict = 250,
        temperature = 0.85,
        top_p = 0.9
      )
      
      result <- sub(".*Example:", "", example)
      cat("📝 AI Example:\n", trimws(result), "\n\n")
    } else {
      cat("📝 This exercise would generate a creative example\n\n")
    }
    
    if (interactive()) Sys.sleep(1)
  }
  
  if (!is.null(ctx)) {
    edge_free_model(ctx)
  }
}

# Main demo function
demo_creative_writing <- function() {
  cat("🎬 Creative Writing Demo\n")
  cat(rep("-", 40), "\n\n")
  
  ctx <- setup_creative_writer()
  
  # Story generation demo
  cat("📖 Story Generation Demo\n")
  story <- generate_story(ctx, "science fiction", "first contact with aliens", "short")
  cat("Generated Story:\n", story, "\n\n")
  
  Sys.sleep(2)
  
  # Poetry demo
  cat("🎭 Poetry Generation Demo\n") 
  poem <- generate_poem(ctx, "romantic", "autumn leaves", "sonnet")
  cat("Generated Poem:\n", poem, "\n\n")
  
  Sys.sleep(2)
  
  # Character creation demo
  cat("👤 Character Creation Demo\n")
  character <- create_character(ctx, "detective", "cyberpunk city")
  cat("Character Profile:\n", character, "\n\n")
  
  Sys.sleep(2)
  
  # Writing prompts demo
  cat("💡 Writing Prompts Demo\n")
  prompts <- generate_writing_prompts(ctx, 3, "time travel")
  cat("Generated Prompts:\n", prompts, "\n\n")
  
  if (!is.null(ctx)) {
    edge_free_model(ctx)
    cat("✅ Creative writing demo complete!\n")
  }
}

# Main execution
if (interactive()) {
  cat("Choose your creative adventure:\n")
  cat("1. demo_creative_writing() - See all capabilities\n")
  cat("2. interactive_story_builder() - Build a complete story\n") 
  cat("3. creative_workshop() - Writing exercises and techniques\n\n")
  cat("Running demo automatically:\n")
  demo_creative_writing()
} else {
  demo_creative_writing()
}