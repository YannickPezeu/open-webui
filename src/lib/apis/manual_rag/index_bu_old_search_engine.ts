// src/lib/apis/search.ts

import { toast } from 'svelte-sonner';
// Importer i18n si vous l'utilisez pour les toasts. Sinon, utilisez des chaînes de caractères simples.
// import { i18n } from '$lib/stores'; // L'import exact peut varier
// const { t } = i18n;

export const searchDocuments = async (query: string) => {
	if (!query.trim()) {
		// Remplacer $i18n.t par une chaîne ou importer i18n correctement
		toast.info('Please enter a search query in the message box.');
		return;
	}

	const encodedQuery = encodeURIComponent(query);
	const url = `http://localhost:8000/search/LEX_AND_RH/${encodedQuery}`;

	console.log(`Sending search request to: ${url}`);
	toast.info('Searching...');

	try {
		const response = await fetch(url, {
			method: 'GET',
			headers: {
				Accept: 'application/json'
			}
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({
				detail: 'Failed to parse error response from server.'
			}));
			throw new Error(
				`API Error: ${response.status} ${response.statusText} - ${errorData.detail}`
			);
		}

		const data = await response.json();
		console.log('Search results:', data);

		if (data.results && data.results.length > 0) {
			toast.success('Search successful! Results are in the console.');
		} else {
			toast.success('Search complete, but no results were found.');
		}
        
        // Optionnellement, vous pouvez retourner les données pour un traitement ultérieur
        return data;

	} catch (error) {
		console.error('Failed to fetch search results:', error);
		toast.error(`Search failed: ${error.message}`);
        return null; // Retourner null en cas d'erreur
	}
};