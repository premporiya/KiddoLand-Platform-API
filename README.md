# KiddoLand Backend API

AI-powered story generation and rewriting backend for children aged 1-18.

## Project Overview

This is the backend API for the KiddoLand project. It provides story generation and rewriting capabilities using Hugging Face's AI models, with built-in child safety filtering.

### Key Features

- ✅ Free-form story generation from user prompts
- ✅ Arbitrary story rewriting based on user instructions
- ✅ Age-appropriate content (1-18 years)
- ✅ Child safety filtering
- ✅ No database (in-memory processing)
- ✅ Clean REST API for frontend integration

## Tech Stack

- **Language**: Python 3.9+
- **Framework**: FastAPI
- **AI Provider**: Hugging Face Inference API
- **Environment**: dotenv for configuration

## Project Structure

```
kiddoland_API/
├── main.py                          # Application entry point
├── routers/
│   └── story.py                     # Story endpoints (generate, rewrite)
├── utils/
│   ├── huggingface_client.py        # Hugging Face API integration
│   └── safety_filter.py             # Child safety filtering
├── schemas/
│   └── story.py                     # Pydantic request/response models
├── .env                             # Environment variables (you need to create this)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Cleanup and Optimization Summary

The FastAPI project structure was reviewed to separate runtime-critical files from local or auto-generated artifacts. The cleanup focused on repository hygiene and maintainability, keeping the codebase ready for version control and deployment. No core application code was modified or deleted.

### Essential Runtime Files (and Why)

- `main.py`: FastAPI application entry point and server startup.
- `routers/`: API route handlers used by the application.
- `schemas/`: Pydantic request/response models used for validation and documentation.
- `utils/`: Shared utilities required by routes (auth, AI client, safety checks).
- `requirements.txt`: Dependency list needed to install and run the API.
- `.env`: Local environment configuration for API tokens and settings.

### Removed or Safe-to-Remove Artifacts (and Why)

- `.venv/`: Local virtual environment; environment-specific and re-creatable.
- `__pycache__/`: Python bytecode cache; auto-generated on run.
- `routers/__pycache__/`: Python bytecode cache; auto-generated on run.
- `schemas/__pycache__/`: Python bytecode cache; auto-generated on run.
- `utils/__pycache__/`: Python bytecode cache; auto-generated on run.
- `.idea/`: IDE metadata; not required to run the API.
- `.github/appmod/` and `.github/appmod/appcat/`: Empty folders; not used by the API.

### How the Cleanup Was Done

- Reviewed the project tree and module imports to identify runtime dependencies.
- Classified local, auto-generated, and IDE-specific artifacts as non-runtime.
- Removed or flagged non-runtime artifacts for removal, while leaving core application files intact.

## Setup Instructions

### 1. Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Hugging Face account (free)

### 2. Get Hugging Face API Token

1. Go to [Hugging Face](https://huggingface.co/)
2. Create a free account or log in
3. Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
4. Click "New token"
5. Give it a name (e.g., "KiddoLand")
6. Select "Read" permission
7. Copy the token

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Required Environment Variables

Create a file named `.env` in the project root with the following content:

```
HUGGINGFACE_API_TOKEN=hf_your_token_here
HUGGINGFACE_API_URL=https://router.huggingface.co/v1/chat/completions
HUGGINGFACE_MODEL=mistralai/Mistral-7B-Instruct-v0.2

KIDDOLAND_AUTH_SECRET=change_me
KIDDOLAND_AUTH_TTL_SECONDS=3600
KIDDOLAND_AUTH_USERS=[{"email":"parent@kiddoland.local","password":"Parent123!","role":"Parent","modes":["home"]}]

MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/
MONGODB_DB_NAME=kiddoland
MONGODB_USERS_COLLECTION=users
```

Replace `hf_your_token_here` with your actual Hugging Face token from step 2. The `KIDDOLAND_AUTH_USERS` value is optional; if omitted, demo users are created at runtime.

### 5. Run the Server

**Option 1: Run directly in PyCharm (Easiest)**

- Right-click on `main.py`
- Select "Run 'main'"
- The server will start automatically

**Option 2: Run via terminal**

```bash
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

### 6. Access API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

In Swagger, click **Authorize** and enter:

```
Bearer <API_TOKEN>
```

Use the same value you set in `.env`.

### 7. Verify Hugging Face Integration

Start the server, then run the verification script:

```bash
python scripts/verify_hf_integration.py
```

Optional overrides:

```bash
set KIDDOLAND_API_BASE_URL=http://127.0.0.1:8000
set KIDDOLAND_TEST_EMAIL=parent@kiddoland.local
set KIDDOLAND_TEST_PASSWORD=Parent123!
set KIDDOLAND_TEST_MODE=home
set KIDDOLAND_TEST_AGE=8
set KIDDOLAND_TEST_PROMPT=Write a short story about a friendly robot
```

## API Endpoints

### 1. Health Check

```
GET /
GET /health
```

**Response:**

```json
{
  "status": "online",
  "service": "KiddoLand API",
  "version": "1.0.0"
}
```

### 2. Generate Story

```
POST /story/generate
```

**Request Body:**

```json
{
  "age": 10,
  "prompt": "Write a story about a shy dragon who learns to make friends"
}
```

**Response:**

```json
{
  "story": "Once upon a time, in a misty mountain range, there lived a shy dragon named Ember..."
}
```

**Parameters:**

- `age` (integer, required): Child's age, must be between 1-18
- `prompt` (string, required): Free-form story prompt (any text, 1-2000 characters)

### 3. Rewrite Story

```
POST /story/rewrite
```

**Request Body:**

```json
{
  "age": 10,
  "original_story": "Once upon a time, there was a shy dragon...",
  "instruction": "Change the middle part to make it funnier"
}
```

**Response:**

```json
{
  "story": "Once upon a time, there was a shy dragon who was terrified of sneezing because every time he did, bubbles came out instead of fire..."
}
```

**Parameters:**

- `age` (integer, required): Child's age, must be between 1-18
- `original_story` (string, required): The original story text (1-10000 characters)
- `instruction` (string, required): Free-form rewrite instruction (1-1000 characters)

**Rewrite Instruction Examples:**

- "Change the ending"
- "Make it happier"
- "Make it darker"
- "Add a new character"
- "Remove the villain"
- "Make the middle funnier"
- "Change the entire story to be about space"

## Frontend Integration Guide

### Base URL

```
http://localhost:8000
```

### Example: Generate Story (JavaScript/Fetch)

```javascript
async function generateStory(age, prompt) {
  const response = await fetch("http://localhost:8000/story/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.API_TOKEN}`,
    },
    body: JSON.stringify({
      age: age,
      prompt: prompt,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  return data.story;
}

// Usage
const story = await generateStory(10, "Write a story about a brave knight");
console.log(story);
```

### Example: Rewrite Story (JavaScript/Fetch)

```javascript
async function rewriteStory(age, originalStory, instruction) {
  const response = await fetch("http://localhost:8000/story/rewrite", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.API_TOKEN}`,
    },
    body: JSON.stringify({
      age: age,
      original_story: originalStory,
      instruction: instruction,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }

  const data = await response.json();
  return data.story;
}

// Usage
const rewritten = await rewriteStory(
  10,
  "Once upon a time...",
  "Make it happier",
);
console.log(rewritten);
```

### Example: Using Axios

```javascript
import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

// Generate story
const generateStory = async (age, prompt) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/story/generate`,
      {
        age,
        prompt,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.API_TOKEN}`,
        },
      },
    );
    return response.data.story;
  } catch (error) {
    console.error("Error:", error.response?.data?.detail || error.message);
    throw error;
  }
};

// Rewrite story
const rewriteStory = async (age, originalStory, instruction) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/story/rewrite`,
      {
        age,
        original_story: originalStory,
        instruction,
      },
      {
        headers: {
          Authorization: `Bearer ${process.env.API_TOKEN}`,
        },
      },
    );
    return response.data.story;
  } catch (error) {
    console.error("Error:", error.response?.data?.detail || error.message);
    throw error;
  }
};
```

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error (AI model error, network error, etc.)

**Error Response Format:**

```json
{
  "detail": "Error message here"
}
```

**Frontend Error Handling:**

```javascript
try {
  const story = await generateStory(10, prompt);
  // Success - use the story
} catch (error) {
  if (error.response) {
    // API returned an error
    console.error("API Error:", error.response.data.detail);
    // Show error to user
  } else {
    // Network error
    console.error("Network Error:", error.message);
  }
}
```

## Child Safety Features

The backend includes a soft safety filter that checks for:

- ❌ Extreme violence
- ❌ Explicit sexual content
- ❌ Strong profanity
- ❌ Harmful content

If unsafe content is detected, the API returns a polite refusal message instead of the generated content.

**Note**: The filter is intentionally SOFT to avoid over-restricting creative storytelling. Normal story elements (adventure, conflict, fantasy) are allowed.

## Important Notes for Frontend Team

1. **CORS**: CORS is enabled for all origins in development. Configure the allowed origins in `main.py` for production.

2. **Response Times**: Story generation may take 5-30 seconds depending on:
   - Hugging Face model load time
   - Story complexity
   - API traffic

   **Recommendation**: Show a loading indicator in your UI.

3. **Model Loading**: On first request, the Hugging Face model may need to load. This can take up to 60 seconds. Subsequent requests will be faster.

4. **Rate Limiting**: Hugging Face free tier has rate limits. For production, consider:
   - Upgrading to Hugging Face Pro
   - Using a different AI provider
   - Implementing request queuing

5. **Authentication**: Endpoints require a Bearer token. Set `API_TOKEN` and pass it via the `Authorization: Bearer <token>` header.

6. **No Database**: Stories are not saved. Implement frontend storage (localStorage, session storage) if persistence is needed before saving to your frontend's database.

## Testing with curl

### Generate Story

```bash
curl -X POST "http://localhost:8000/story/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change_me" \
  -d "{\"age\": 10, \"prompt\": \"Write a story about a brave knight\"}"
```

### Rewrite Story

```bash
curl -X POST "http://localhost:8000/story/rewrite" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change_me" \
  -d "{\"age\": 10, \"original_story\": \"Once upon a time...\", \"instruction\": \"Make it funnier\"}"
```

## Troubleshooting

**"HUGGINGFACE_API_TOKEN not found"**

- Make sure you created `.env` file and added your token

**"Model is loading" error**

- Wait 30-60 seconds and try again. First request loads the model

**Slow responses**

- Normal for Hugging Face free tier

**"Network error"**

- Check your internet connection

## Production Deployment

For production deployment:

1. **Change CORS settings** in `main.py` to only allow your frontend domain
2. **Use environment variables** for configuration (not .env files)
3. **Add authentication** if needed
4. **Set up monitoring** and logging
5. **Use a production ASGI server** (already using uvicorn)
6. **Consider Hugging Face Pro** for better performance and rate limits

## Support

For issues or questions:

- Check the interactive API docs at `/docs`
- Review this README
- Contact the backend developer (you!)

---

**Built with ❤️ for KiddoLand Academic Project**
