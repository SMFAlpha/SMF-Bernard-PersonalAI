// server.js
/*
 * Bernard Voice Chat App - Node.js Backend
 * This server aims to replicate the functionality of the Python Flask application (v1_5o)
 * with optimizations for latency, particularly in context retrieval for chat completions.
 * Now includes support for PDF and DOCX file loading in the knowledge base.
 */

// Core Modules
const http = require('http');
const path = require('path');
const fs = require('fs-extra');
const { v4: uuidv4 } = require('uuid');
const { performance } = require('perf_hooks');

// External Dependencies
const express = require('express');
const { Server } = require('socket.io');
const dotenv = require('dotenv');
const multer = require('multer');
const OpenAI = require('openai');
const Groq = require('groq-sdk');
const axios = require('axios');
const FormData = require('form-data');
const { HierarchicalNSW } = require('hnswlib-node');
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');

// Load environment variables
dotenv.config();

// --- Configuration Constants ---
const STATIC_FOLDER = path.join(__dirname, 'static');
const AUDIO_FOLDER = path.join(STATIC_FOLDER, 'temp_audio');
const DATA_DIR = path.join(__dirname, 'data');
const KB_DIR = process.env.KB_DIR || path.join(DATA_DIR, 'kb');
const VECTOR_STORE_PARENT_DIR = process.env.VECTOR_STORE_DIR || path.join(DATA_DIR, 'vector_store');
const KB_VECTOR_STORE_DIR = path.join(VECTOR_STORE_PARENT_DIR, 'kb_store');
const CHAT_VECTOR_STORE_DIR = path.join(VECTOR_STORE_PARENT_DIR, 'chat_store');
const SUBJECTS_FILE = process.env.SUBJECTS_FILE || path.join(DATA_DIR, 'subjects.json');
const CHAT_HISTORY_JSON = process.env.CHAT_HISTORY_JSON || path.join(DATA_DIR, 'chat_history.json');
const PHRASES_FILE = process.env.PHRASES_FILE || path.join(DATA_DIR, 'phrases.json');
const SETTINGS_FILE = process.env.SETTINGS_FILE || path.join(DATA_DIR, 'settings.json');

// Ensure necessary directories exist
fs.ensureDirSync(STATIC_FOLDER);
fs.ensureDirSync(AUDIO_FOLDER);
fs.ensureDirSync(KB_DIR);
fs.ensureDirSync(KB_VECTOR_STORE_DIR);
fs.ensureDirSync(CHAT_VECTOR_STORE_DIR);
fs.ensureDirSync(path.dirname(SUBJECTS_FILE));
fs.ensureDirSync(path.dirname(CHAT_HISTORY_JSON));
fs.ensureDirSync(path.dirname(PHRASES_FILE));
fs.ensureDirSync(path.dirname(SETTINGS_FILE));

// --- API Keys and Client Initialization ---
let OPENAI_API_KEY = process.env.OPENAI_API_KEY;
let ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
let GROQ_API_KEY = process.env.GROQ_API_KEY;
let PERPLEXITY_API_KEY = process.env.PERPLEXITY_API_KEY;
const PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions";

let openai;
let groq;

// --- Global State ---
let currentSettings = {};
let CURRENT_SUBJECT = ["Default Subject"];
let kbVectorStore;
let chatVectorStore;
const sessionContextQueues = {};
const sessionTranscriptQueues = {};

// --- Express App and Middleware ---
const app = express();
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(express.static(STATIC_FOLDER));

const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, KB_DIR),
    filename: (req, file, cb) => cb(null, file.originalname)
});
const audioUploadStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, AUDIO_FOLDER),
    filename: (req, file, cb) => cb(null, `upload_${uuidv4()}_${file.originalname}`)
});

const docUpload = multer({ storage: storage });
const audioUpload = multer({ storage: audioUploadStorage });

// --- Server and Socket.IO Setup ---
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*", methods: ["GET", "POST"] } });

// --- VectorStore Class ---
class LocalVectorStore {
    constructor(dimension = 1536, space = 'cosine') {
        this.dimension = dimension;
        this.space = space;
        this.index = new HierarchicalNSW(this.space, this.dimension);
        this.documents = [];
        this.isInitialized = false;
    }

    async initializeIndex(maxElements = 10000) {
        if (!this.isInitialized) {
            this.index.initIndex(maxElements);
            this.isInitialized = true;
        }
    }

    async _getEmbeddings(texts, batchSize = 20) {
        if (!openai) throw new Error("OpenAI client not initialized for embeddings.");
        const allEmbeddings = [];
        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            try {
                const response = await openai.embeddings.create({
                    model: 'text-embedding-ada-002',
                    input: batch
                });
                allEmbeddings.push(...response.data.map(item => item.embedding));
            } catch (error) {
                console.error('Error getting embeddings for batch:', error.response?.data || error.message);
                allEmbeddings.push(...Array(batch.length).fill(Array(this.dimension).fill(0)));
            }
        }
        return allEmbeddings;
    }

    async addDocuments(docs) {
        if (!docs || docs.length === 0) return;
        if (!this.isInitialized) await this.initializeIndex(this.documents.length + docs.length + 1000);

        const texts = docs.map(doc => doc.pageContent);
        const embeddings = await this._getEmbeddings(texts);

        for (let i = 0; i < embeddings.length; i++) {
            if (embeddings[i].length !== this.dimension) {
                console.warn(`Skipping document due to embedding dimension mismatch. Expected ${this.dimension}, got ${embeddings[i].length}. Text: ${texts[i].substring(0,50)}...`);
                continue;
            }
            const newIndexLabel = this.documents.length;
            this.index.addPoint(embeddings[i], newIndexLabel);
            this.documents.push({
                pageContent: texts[i],
                metadata: docs[i].metadata || {}
            });
        }
    }

    async similaritySearch(query, k = 3, similarityThreshold = 0.2) {
        if (!this.isInitialized || this.documents.length === 0) return [];
        const queryEmbedding = (await this._getEmbeddings([query]))[0];
        if (queryEmbedding.length !== this.dimension) {
            console.warn(`Query embedding dimension mismatch. Expected ${this.dimension}, got ${queryEmbedding.length}.`);
            return [];
        }

        const result = this.index.searchKnn(queryEmbedding, k);
        const { neighbors, distances } = result;

        const foundDocs = [];
        for (let i = 0; i < neighbors.length; i++) {
            const docIndex = neighbors[i];
            const similarity = this.space === 'l2' ? 1 / (1 + distances[i]) : 1 - distances[i];

            if (similarity >= similarityThreshold && this.documents[docIndex]) {
                foundDocs.push({
                    ...this.documents[docIndex],
                    metadata: { ...this.documents[docIndex].metadata, score: similarity }
                });
            }
        }
        return foundDocs.sort((a, b) => b.metadata.score - a.metadata.score);
    }

    async save(directory) {
        await fs.ensureDir(directory);
        const indexPath = path.join(directory, 'index.bin');
        const docsPath = path.join(directory, 'documents.json');
        this.index.writeIndexSync(indexPath);
        await fs.writeJson(docsPath, {
            dimension: this.dimension,
            space: this.space,
            documents: this.documents,
            isInitialized: this.isInitialized
        });
        console.log(`Vector store saved to ${directory} with ${this.documents.length} documents.`);
    }

    static async load(directory, dimension = 1536, space = 'cosine') {
        const indexPath = path.join(directory, 'index.bin');
        const docsPath = path.join(directory, 'documents.json');
        if (!await fs.pathExists(indexPath) || !await fs.pathExists(docsPath)) {
            console.warn(`Index or documents file not found in ${directory}. Creating new store.`);
            const newStore = new LocalVectorStore(dimension, space);
            await newStore.initializeIndex();
            return newStore;
        }

        const store = new LocalVectorStore(dimension, space);
        const savedData = await fs.readJson(docsPath);
        store.documents = savedData.documents;
        store.isInitialized = savedData.isInitialized;
        if (store.isInitialized) {
            store.index.readIndexSync(indexPath);
        }
        return store;
    }
}

// --- Utility Functions ---
async function loadSettings() {
    try {
        if (!await fs.pathExists(SETTINGS_FILE)) {
            const defaultSettings = createDefaultSettingsStructure();
            await fs.writeJson(SETTINGS_FILE, defaultSettings, { spaces: 2 });
            return defaultSettings;
        }
        return await fs.readJson(SETTINGS_FILE);
    } catch (error) {
        console.error('Error loading settings:', error);
        return createDefaultSettingsStructure();
    }
}

function createDefaultSettingsStructure() {
    return {
        llm: {
            provider: "openai",
            model: "gpt-4-1106-preview",
            temperature: 0.7,
            max_tokens: 150,
            top_p: 1,
            frequency_penalty: 0,
            presence_penalty: 0
        },
        voice: {
            provider: "elevenlabs",
            voice_id: "pNInz6obpgDQGcFmaJgB",
            stability: 0.5,
            similarity_boost: 0.75
        },
        retrieval: {
            kb_results: 2,
            chat_results: 2,
            days_lookback: 7
        }
    };
}

async function saveSettingsToFile(settingsToSave) {
    try {
        await fs.writeJson(SETTINGS_FILE, settingsToSave, { spaces: 2 });
        currentSettings = settingsToSave;
        return { success: true };
    } catch (error) {
        console.error('Error saving settings:', error);
        return { success: false, error: error.message };
    }
}

function getSystemPrompt() {
    return "You are Bernard, a highly intelligent and empathetic AI assistant. You engage in natural, flowing conversations while maintaining professionalism. You're direct and concise in your responses, avoiding unnecessary apologies or hesitation. You have access to past conversations and a knowledge base to provide context-aware responses.";
}

async function loadSubjects() {
    try {
        if (!await fs.pathExists(SUBJECTS_FILE)) {
            await fs.writeJson(SUBJECTS_FILE, ["Default Subject"], { spaces: 2 });
            return ["Default Subject"];
        }
        const subjects = await fs.readJson(SUBJECTS_FILE);
        return Array.isArray(subjects) ? subjects : ["Default Subject"];
    } catch (error) {
        console.error('Error loading subjects:', error);
        return ["Default Subject"];
    }
}

async function saveSubjects(subjects) {
    try {
        await fs.writeJson(SUBJECTS_FILE, subjects, { spaces: 2 });
        return { success: true };
    } catch (error) {
        console.error('Error saving subjects:', error);
        return { success: false, error: error.message };
    }
}

async function speakTextEleven(textToSpeak) {
    if (!ELEVENLABS_API_KEY) {
        throw new Error("ElevenLabs API key not configured");
    }

    const url = `https://api.elevenlabs.io/v1/text-to-speech/${currentSettings.voice.voice_id}`;
    const headers = {
        'Accept': 'audio/mpeg',
        'Content-Type': 'application/json',
        'xi-api-key': ELEVENLABS_API_KEY
    };

    const data = {
        text: textToSpeak,
        model_id: "eleven_monolingual_v1",
        voice_settings: {
            stability: currentSettings.voice.stability,
            similarity_boost: currentSettings.voice.similarity_boost
        }
    };

    try {
        const response = await axios({
            method: 'post',
            url: url,
            headers: headers,
            data: data,
            responseType: 'arraybuffer'
        });

        const outputPath = path.join(AUDIO_FOLDER, `response_${uuidv4()}.mp3`);
        await fs.writeFile(outputPath, response.data);
        return outputPath;
    } catch (error) {
        console.error('Error in text-to-speech:', error.response?.data || error.message);
        throw error;
    }
}

async function transcribeAudioEleven(audioFilePath) {
    if (!ELEVENLABS_API_KEY) {
        throw new Error("ElevenLabs API key not configured");
    }

    const url = 'https://api.elevenlabs.io/v1/speech-to-text';
    const formData = new FormData();
    formData.append('audio', fs.createReadStream(audioFilePath));
    
    try {
        const response = await axios.post(url, formData, {
            headers: {
                ...formData.getHeaders(),
                'xi-api-key': ELEVENLABS_API_KEY
            }
        });

        return response.data.text;
    } catch (error) {
        console.error('Error in speech-to-text:', error.response?.data || error.message);
        throw error;
    }
}

function cleanupTranscript(transcript) {
    return transcript
        .replace(/^\s+|\s+$/g, '')
        .replace(/\s+/g, ' ')
        .replace(/[.,!?;:]\s*/g, match => match.trim() + ' ')
        .replace(/\s+([.,!?;:])/g, '$1')
        .replace(/\s+'/g, "'")
        .replace(/'\s+/g, "'")
        .replace(/\s+"/g, '"')
        .replace(/"\s+/g, '"')
        .trim();
}

async function getOpenAICompletion(messages, nCompletions = 1, stream = false, socketForStream, sessionIdForStream) {
    if (!openai) {
        openai = new OpenAI({ apiKey: OPENAI_API_KEY });
    }

    const params = {
        model: currentSettings.llm.model,
        messages: messages,
        temperature: currentSettings.llm.temperature,
        max_tokens: currentSettings.llm.max_tokens,
        top_p: currentSettings.llm.top_p,
        frequency_penalty: currentSettings.llm.frequency_penalty,
        presence_penalty: currentSettings.llm.presence_penalty,
        n: nCompletions,
        stream: stream
    };

    try {
        if (stream) {
            const stream = await openai.chat.completions.create(params);
            let fullResponse = '';
            
            for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content || '';
                if (content) {
                    fullResponse += content;
                    socketForStream.emit('stream_response', {
                        sessionId: sessionIdForStream,
                        chunk: content
                    });
                }
            }
            
            return [{ content: fullResponse }];
        } else {
            const response = await openai.chat.completions.create(params);
            return response.choices.map(choice => ({ content: choice.message.content }));
        }
    } catch (error) {
        console.error('Error in OpenAI completion:', error.response?.data || error.message);
        throw error;
    }
}

async function getGroqCompletion(messages, nCompletions = 1, stream = false, socketForStream, sessionIdForStream) {
    if (!groq) {
        groq = new Groq({ apiKey: GROQ_API_KEY });
    }

    const params = {
        model: currentSettings.llm.model,
        messages: messages,
        temperature: currentSettings.llm.temperature,
        max_tokens: currentSettings.llm.max_tokens,
        top_p: currentSettings.llm.top_p,
        stream: stream
    };

    try {
        if (stream) {
            const stream = await groq.chat.completions.create(params);
            let fullResponse = '';
            
            for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content || '';
                if (content) {
                    fullResponse += content;
                    socketForStream.emit('stream_response', {
                        sessionId: sessionIdForStream,
                        chunk: content
                    });
                }
            }
            
            return [{ content: fullResponse }];
        } else {
            const response = await groq.chat.completions.create(params);
            return response.choices.map(choice => ({ content: choice.message.content }));
        }
    } catch (error) {
        console.error('Error in Groq completion:', error.response?.data || error.message);
        throw error;
    }
}

async function getPerplexityResponse(query) {
    try {
        const response = await axios.post(PERPLEXITY_API_URL, {
            model: "mistral-7b-instruct",
            messages: [{ role: "user", content: query }]
        }, {
            headers: {
                'Authorization': `Bearer ${PERPLEXITY_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });
        return response.data.choices[0].message.content;
    } catch (error) {
        console.error('Error in Perplexity API call:', error.response?.data || error.message);
        throw error;
    }
}

async function saveChatExchangeJson(userText, bernardText) {
    try {
        const timestamp = new Date().toISOString();
        const exchange = {
            timestamp: timestamp,
            subject: CURRENT_SUBJECT[0],
            exchange: {
                user: userText,
                bernard: bernardText
            }
        };

        let chatHistory = [];
        if (await fs.pathExists(CHAT_HISTORY_JSON)) {
            chatHistory = await fs.readJson(CHAT_HISTORY_JSON);
        }

        chatHistory.push(exchange);
        await fs.writeJson(CHAT_HISTORY_JSON, chatHistory, { spaces: 2 });
        return true;
    } catch (error) {
        console.error('Error saving chat exchange:', error);
        return false;
    }
}

async function loadTypedDocuments() {
    const documents = [];
    const files = await fs.readdir(KB_DIR);
    
    for (const file of files) {
        const filePath = path.join(KB_DIR, file);
        const stats = await fs.stat(filePath);
        if (!stats.isFile()) continue;

        const fileExt = path.extname(file).toLowerCase();
        let content = '';
        let metadata = {
            source: file,
            type: fileExt.slice(1),
            created: stats.birthtime,
            modified: stats.mtime
        };

        try {
            switch (fileExt) {
                case '.txt':
                case '.md':
                    content = await fs.readFile(filePath, 'utf8');
                    break;
                    
                case '.pdf':
                    const pdfData = await fs.readFile(filePath);
                    const pdfResult = await pdfParse(pdfData);
                    content = pdfResult.text;
                    metadata = {
                        ...metadata,
                        pages: pdfResult.numpages,
                        info: pdfResult.info
                    };
                    break;
                    
                case '.docx':
                    const docxResult = await mammoth.extractRawText({
                        path: filePath
                    });
                    content = docxResult.value;
                    break;
                    
                case '.json':
                case '.jsonl':
                    const jsonContent = await fs.readFile(filePath, 'utf8');
                    const lines = jsonContent.trim().split('\n');
                    for (const line of lines) {
                        try {
                            const entry = JSON.parse(line);
                            if (typeof entry === 'object') {
                                const entryContent = Object.values(entry).join(' ');
                                documents.push({
                                    pageContent: entryContent,
                                    metadata: {
                                        ...metadata,
                                        original: entry
                                    }
                                });
                            }
                        } catch (e) {
                            console.warn(`Error parsing JSON line in ${file}:`, e.message);
                        }
                    }
                    continue;
                    
                default:
                    console.warn(`Unsupported file type: ${fileExt}`);
                    continue;
            }

            if (content) {
                const chunks = chunkText(content);
                chunks.forEach((chunk, index) => {
                    documents.push({
                        pageContent: chunk,
                        metadata: {
                            ...metadata,
                            chunk: index + 1,
                            totalChunks: chunks.length
                        }
                    });
                });
            }
        } catch (error) {
            console.error(`Error processing file ${file}:`, error);
        }
    }
    
    return documents;
}

function chunkText(text, chunkSize = 1000, chunkOverlap = 100) {
    const chunks = [];
    let startIndex = 0;
    
    while (startIndex < text.length) {
        let endIndex = startIndex + chunkSize;
        let chunk = text.slice(startIndex, endIndex);
        
        // If we're not at the end of the text, try to break at a sentence boundary
        if (endIndex < text.length) {
            const lastPeriod = chunk.lastIndexOf('.');
            const lastQuestion = chunk.lastIndexOf('?');
            const lastExclamation = chunk.lastIndexOf('!');
            const lastBreak = Math.max(lastPeriod, lastQuestion, lastExclamation);
            
            if (lastBreak > chunkSize * 0.5) {
                endIndex = startIndex + lastBreak + 1;
                chunk = text.slice(startIndex, endIndex);
            }
        }
        
        chunks.push(chunk.trim());
        startIndex = endIndex - chunkOverlap;
    }
    
    return chunks;
}

async function buildAndSaveKbVectorStore() {
    console.log('Building knowledge base vector store...');
    const startTime = performance.now();
    
    try {
        const documents = await loadTypedDocuments();
        console.log(`Loaded ${documents.length} document chunks`);
        
        kbVectorStore = new LocalVectorStore();
        await kbVectorStore.initializeIndex(documents.length + 1000);
        await kbVectorStore.addDocuments(documents);
        await kbVectorStore.save(KB_VECTOR_STORE_DIR);
        
        const endTime = performance.now();
        console.log(`KB vector store built and saved in ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
        return true;
    } catch (error) {
        console.error('Error building KB vector store:', error);
        return false;
    }
}

async function loadKbVectorStore() {
    try {
        kbVectorStore = await LocalVectorStore.load(KB_VECTOR_STORE_DIR);
        console.log(`Loaded KB vector store with ${kbVectorStore.documents.length} documents`);
        return true;
    } catch (error) {
        console.error('Error loading KB vector store:', error);
        return false;
    }
}

async function buildChatVectorStore(incremental = false) {
    console.log('Building chat history vector store...');
    const startTime = performance.now();
    
    try {
        let chatHistory = [];
        if (await fs.pathExists(CHAT_HISTORY_JSON)) {
            chatHistory = await fs.readJson(CHAT_HISTORY_JSON);
        }

        if (chatHistory.length === 0) {
            console.log('No chat history found');
            return false;
        }

        const documents = [];
        for (const entry of chatHistory) {
            const exchange = entry.exchange;
            const combinedText = `User: ${exchange.user}\nBernard: ${exchange.bernard}`;
            documents.push({
                pageContent: combinedText,
                metadata: {
                    timestamp: entry.timestamp,
                    subject: entry.subject
                }
            });
        }

        if (incremental && chatVectorStore?.documents) {
            const existingDocs = chatVectorStore.documents;
            const newDocs = documents.slice(existingDocs.length);
            if (newDocs.length > 0) {
                await chatVectorStore.addDocuments(newDocs);
            }
        } else {
            chatVectorStore = new LocalVectorStore();
            await chatVectorStore.initializeIndex(documents.length + 1000);
            await chatVectorStore.addDocuments(documents);
        }

        await chatVectorStore.save(CHAT_VECTOR_STORE_DIR);
        
        const endTime = performance.now();
        console.log(`Chat vector store built and saved in ${((endTime - startTime) / 1000).toFixed(2)} seconds`);
        return true;
    } catch (error) {
        console.error('Error building chat vector store:', error);
        return false;
    }
}

async function loadChatVectorStore() {
    try {
        chatVectorStore = await LocalVectorStore.load(CHAT_VECTOR_STORE_DIR);
        console.log(`Loaded chat vector store with ${chatVectorStore.documents.length} documents`);
        return true;
    } catch (error) {
        console.error('Error loading chat vector store:', error);
        return false;
    }
}

async function retrieveKbContext(query, k = 2, similarityThreshold = 0.1) {
    if (!kbVectorStore) return [];
    const results = await kbVectorStore.similaritySearch(query, k, similarityThreshold);
    return results.map(doc => ({
        content: doc.pageContent,
        metadata: doc.metadata
    }));
}

async function retrieveChatHistoryContext(query, subject = null, days = null, maxExchanges = 2, similarityThreshold = 0.3) {
    if (!chatVectorStore) return [];
    const results = await chatVectorStore.similaritySearch(query, maxExchanges * 2, similarityThreshold);
    
    return results
        .filter(doc => {
            if (subject && doc.metadata.subject !== subject) return false;
            if (days) {
                const cutoff = new Date();
                cutoff.setDate(cutoff.getDate() - days);
                return new Date(doc.metadata.timestamp) >= cutoff;
            }
            return true;
        })
        .slice(0, maxExchanges)
        .map(doc => ({
            content: doc.pageContent,
            metadata: doc.metadata
        }));
}

function addToSessionTranscriptQueue(sessionId, transcript) {
    if (!sessionTranscriptQueues[sessionId]) {
        sessionTranscriptQueues[sessionId] = [];
    }
    
    sessionTranscriptQueues[sessionId].push(transcript);
    
    // Keep only the last 5 transcripts
    if (sessionTranscriptQueues[sessionId].length > 5) {
        sessionTranscriptQueues[sessionId].shift();
    }
}

function addToSessionContextQueue(sessionId, userText, bernardText) {
    if (!sessionContextQueues[sessionId]) {
        sessionContextQueues[sessionId] = [];
    }
    
    sessionContextQueues[sessionId].push({
        user: userText,
        bernard: bernardText
    });
    
    // Keep only the last 5 exchanges
    if (sessionContextQueues[sessionId].length > 5) {
        sessionContextQueues[sessionId].shift();
    }
}

async function getFullConversationContext(sessionId, currentQuery) {
    const contextParts = [];
    
    // Get relevant KB context
    const kbResults = await retrieveKbContext(
        currentQuery,
        currentSettings.retrieval.kb_results,
        0.1
    );
    
    if (kbResults.length > 0) {
        contextParts.push("Relevant knowledge base information:");
        kbResults.forEach(result => {
            contextParts.push(`[Source: ${result.metadata.source}] ${result.content}`);
        });
    }
    
    // Get relevant chat history context
    const chatResults = await retrieveChatHistoryContext(
        currentQuery,
        CURRENT_SUBJECT[0],
        currentSettings.retrieval.days_lookback,
        currentSettings.retrieval.chat_results,
        0.3
    );
    
    if (chatResults.length > 0) {
        contextParts.push("\nRelevant conversation history:");
        chatResults.forEach(result => {
            contextParts.push(result.content);
        });
    }
    
    // Add recent session context
    if (sessionContextQueues[sessionId]?.length > 0) {
        contextParts.push("\nRecent conversation in this session:");
        sessionContextQueues[sessionId].forEach(exchange => {
            contextParts.push(`User: ${exchange.user}\nBernard: ${exchange.bernard}`);
        });
    }
    
    // Add recent transcripts if any
    if (sessionTranscriptQueues[sessionId]?.length > 0) {
        contextParts.push("\nRecent transcripts in this session:");
        sessionTranscriptQueues[sessionId].forEach(transcript => {
            contextParts.push(`User said: ${transcript}`);
        });
    }
    
    return contextParts.join("\n\n");
}

// Initialize server
const PORT = process.env.PORT || 3000;

// Socket.IO event handlers
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    let currentSessionId = null;

    socket.on('start_session', async (data) => {
        currentSessionId = data.sessionId || uuidv4();
        socket.emit('session_started', { sessionId: currentSessionId });
    });

    socket.on('end_session', () => {
        if (currentSessionId) {
            delete sessionContextQueues[currentSessionId];
            delete sessionTranscriptQueues[currentSessionId];
            currentSessionId = null;
        }
    });

    socket.on('get_completion', async (data) => {
        try {
            const { text, sessionId, stream = false } = data;
            currentSessionId = sessionId;

            const context = await getFullConversationContext(sessionId, text);
            const messages = [
                { role: "system", content: getSystemPrompt() },
                { role: "user", content: `Context:\n${context}\n\nCurrent message:\n${text}` }
            ];

            let completion;
            if (currentSettings.llm.provider === "openai") {
                completion = await getOpenAICompletion(messages, 1, stream, socket, sessionId);
            } else if (currentSettings.llm.provider === "groq") {
                completion = await getGroqCompletion(messages, 1, stream, socket, sessionId);
            } else {
                throw new Error(`Unknown LLM provider: ${currentSettings.llm.provider}`);
            }

            const bernardResponse = completion[0].content;
            
            if (!stream) {
                socket.emit('completion', {
                    sessionId: sessionId,
                    response: bernardResponse
                });
            } else {
                socket.emit('stream_end', {
                    sessionId: sessionId
                });
            }

            // Save to context queue and chat history
            addToSessionContextQueue(sessionId, text, bernardResponse);
            await saveChatExchangeJson(text, bernardResponse);
            await buildChatVectorStore(true);

        } catch (error) {
            console.error('Error in completion:', error);
            socket.emit('error', {
                message: 'Error generating response',
                details: error.message
            });
        }
    });

    socket.on('transcribe', async (data) => {
        try {
            const { audioPath, sessionId } = data;
            const transcript = await transcribeAudioEleven(audioPath);
            const cleanTranscript = cleanupTranscript(transcript);
            
            addToSessionTranscriptQueue(sessionId, cleanTranscript);
            
            socket.emit('transcription', {
                sessionId: sessionId,
                text: cleanTranscript
            });
            
            // Clean up the audio file
            await fs.unlink(audioPath);
            
        } catch (error) {
            console.error('Error in transcription:', error);
            socket.emit('error', {
                message: 'Error transcribing audio',
                details: error.message
            });
        }
    });

    socket.on('speak', async (data) => {
        try {
            const { text } = data;
            const audioPath = await speakTextEleven(text);
            
            socket.emit('audio_ready', {
                audioPath: path.basename(audioPath)
            });
            
        } catch (error) {
            console.error('Error in text-to-speech:', error);
            socket.emit('error', {
                message: 'Error generating speech',
                details: error.message
            });
        }
    });

    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
        if (currentSessionId) {
            delete sessionContextQueues[currentSessionId];
            delete sessionTranscriptQueues[currentSessionId];
        }
    });
});

// Express routes
app.post('/upload', docUpload.single('file'), async (req, res) => {
    try {
        await buildAndSaveKbVectorStore();
        res.json({ success: true });
    } catch (error) {
        console.error('Error processing uploaded file:', error);
        res.status(500).json({ error: 'Error processing file' });
    }
});

app.post('/upload_audio', audioUpload.single('audio'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No audio file uploaded' });
    }
    res.json({ path: req.file.filename });
});

app.get('/settings', async (req, res) => {
    res.json(currentSettings);
});

app.post('/settings', async (req, res) => {
    const result = await saveSettingsToFile(req.body);
    res.json(result);
});

app.get('/subjects', async (req, res) => {
    const subjects = await loadSubjects();
    res.json(subjects);
});

app.post('/subjects', async (req, res) => {
    const result = await saveSubjects(req.body);
    res.json(result);
});

// Initialize and start server
async function initializeServer() {
    try {
        // Load settings
        currentSettings = await loadSettings();
        
        // Initialize API clients
        if (OPENAI_API_KEY) {
            openai = new OpenAI({ apiKey: OPENAI_API_KEY });
        }
        if (GROQ_API_KEY) {
            groq = new Groq({ apiKey: GROQ_API_KEY });
        }
        
        // Load subjects
        CURRENT_SUBJECT = await loadSubjects();
        
        // Load or build vector stores
        const kbLoaded = await loadKbVectorStore();
        if (!kbLoaded) {
            await buildAndSaveKbVectorStore();
        }
        
        const chatLoaded = await loadChatVectorStore();
        if (!chatLoaded) {
            await buildChatVectorStore();
        }
        
        // Start server
        server.listen(PORT, () => {
            console.log(`Server running on port ${PORT}`);
        });
        
    } catch (error) {
        console.error('Error initializing server:', error);
        process.exit(1);
    }
}

initializeServer(); 