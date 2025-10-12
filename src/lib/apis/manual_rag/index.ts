import { toast } from 'svelte-sonner';
import { WEBUI_API_BASE_URL } from '$lib/constants';

export const searchDocuments = async (
    token: string,
    query: string,
    indexId: string,
    password?: string
) => {
    if (!query.trim()) {
        toast.info('Please enter a search query in the message box.');
        return null;
    }

    let error = null;

    const formData = new FormData();
    formData.append('query', query);
    if (password) {
        formData.append('password', password);
    }

    console.log(`Sending POST request to: ${WEBUI_API_BASE_URL}/libraries/${indexId}/search`);
    // toast.info('Searching...');

    const res = await fetch(`${WEBUI_API_BASE_URL}/libraries/${indexId}/search`, {
        method: 'POST',
        headers: {
            authorization: `Bearer ${token}`
            // Pas de Content-Type pour FormData, le browser le gère
        },
        body: formData
    })
        .then(async (res) => {
            if (!res.ok) throw await res.json();
            return res.json();
        })
        .then((json) => {
            console.log('Search results:', json);
            if (json && json.length > 0) {
                toast.success(`Search successful! Found ${json.length} results.`);
            } else {
                toast.success('Search complete, but no results were found.');
            }
            return json;
        })
        .catch((err) => {
            error = err.detail || err.message || 'Unknown error';
            console.error('Failed to fetch search results:', err);
            toast.error(`Search failed: ${error}`);
            return null;
        });

    if (error) {
        throw error;
    }

    return res;
};


import { generateOpenAIChatCompletion } from '$lib/apis/openai';
import { WEBUI_BASE_URL } from '$lib/constants';

export const generateOptimizedRagQuery = async (
	token: string,
	userQuery: string,
	conversationHistory: any[] = [],
	modelId: string
): Promise<string> => {
	const systemPrompt = `You are a search query optimizer for an EPFL (École Polytechnique Fédérale de Lausanne) document retrieval system. The search engine uses semantic embeddings and reranking to find relevant documents from EPFL's internal knowledge base.

Your task is to transform the user's question into an optimized search query that will retrieve the most relevant EPFL documents.

Context:
- The search engine contains EPFL-specific documents (policies, procedures, research info, administrative docs, etc.)
- It uses semantic search (embeddings) + reranker, so it works better with complete, well-formed questions
- You DON'T need to mention "EPFL" in queries since all documents are already EPFL-specific
- Focus on the actual information need

Rules:
1. Generate a complete, well-formed question in natural language (not keywords)
2. Make the question specific and clear about what information is needed
3. Preserve important context from the conversation if relevant
4. **CRITICAL: Use the SAME language as the user's query. If the language is ambiguous or unclear, default to French.**
5. Keep it concise but complete (typically 10-20 words)
6. Remove conversational filler but keep semantic structure
7. Return ONLY the optimized query, nothing else

Examples:

User: "Can you tell me more about what we discussed earlier regarding overhead rates?"
Context: Previous messages mention "research project funding"
Output: What are the overhead rates for research projects?

User: "Quels sont les principaux avantages ?"
Context: Previous messages discuss "mobilité durable sur le campus"
Output: Quels sont les avantages des solutions de mobilité durable sur le campus ?

User: "How does that work exactly?"
Context: Previous messages mention "PhD student registration process"
Output: How does the PhD student registration process work?

User: "overhead rates"
Context: No previous context
Output: Quels sont les taux d'overhead applicables aux projets de recherche ?

User: "je veux savoir les délais"
Context: Previous messages about "demande de congé académique"
Output: Quels sont les délais pour une demande de congé académique ?

User: "tell me about the policy"
Context: Previous messages discuss "remote work for staff"
Output: What is the remote work policy for staff members?

User: "taux"
Context: Unclear context
Output: Quels sont les taux d'overhead pour les projets ?`;

	const messages = [
		{
			role: 'system',
			content: systemPrompt
		}
	];

	// Ajouter un résumé du contexte si disponible
	if (conversationHistory.length > 0) {
		const recentMessages = conversationHistory
			.slice(-3) // Prendre les 3 derniers messages
			.map((msg) => `${msg.role}: ${msg.content}`)
			.join('\n');

		messages.push({
			role: 'user',
			content: `Conversation context:\n${recentMessages}\n\nCurrent user query: ${userQuery}\n\nGenerate the optimized search query (use the same language as the user query, or French if ambiguous):`
		});
	} else {
		messages.push({
			role: 'user',
			content: `User query: ${userQuery}\n\nGenerate the optimized search query (use the same language as the user query, or French if ambiguous):`
		});
	}

	try {
		const response = await generateOpenAIChatCompletion(
			token,
			{
				model: modelId,
				messages: messages,
				stream: false,
				params: {
					max_tokens: 100,
					temperature: 0.2
				}
			},
			`${WEBUI_BASE_URL}/api`
		);

		if (response && response.choices && response.choices[0]?.message?.content) {
			const optimizedQuery = response.choices[0].message.content.trim();
			console.log('🎯 Optimized query:', optimizedQuery);
			return optimizedQuery;
		}

		// Fallback sur la requête originale si échec
		console.warn('Failed to generate optimized query, using original');
		return userQuery;
	} catch (error) {
		console.error('Error generating optimized query:', error);
		// Fallback sur la requête originale
		return userQuery;
	}
};