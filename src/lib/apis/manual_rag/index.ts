import { toast } from 'svelte-sonner';

export const searchDocuments = async (
    query: string,
    userId: string,
    indexId: string,
    password?: string
) => {
    if (!query.trim()) {
        toast.info('Please enter a search query in the message box.');
        return null;
    }

    // L'URL de votre nouvelle API
    const url = `http://localhost:8000/search/${userId}/${indexId}`;

    // Le corps (body) de la requête POST
    const payload = {
        query: query,
        password: password || null
    };

    console.log(`Sending POST request to: ${url}`);
    toast.info('Searching...');

    try {
        const response = await fetch(url, {
            method: 'POST', // <-- Changement de la méthode
            headers: {
                'Content-Type': 'application/json', // <-- Header nécessaire pour le JSON
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload) // <-- On envoie la requête dans le corps
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({
                detail: 'Failed to parse error response from server.'
            }));
            throw new Error(
                `API Error: ${response.status} ${response.statusText} - ${errorData.detail || 'Unknown error'}`
            );
        }

        // Votre nouvelle API retourne directement un tableau de résultats
        const data = await response.json();
        console.log('Search results:', data);

        if (data && data.length > 0) {
            toast.success(`Search successful! Found ${data.length} results.`);
        } else {
            toast.success('Search complete, but no results were found.');
        }
        
        // On retourne directement le tableau
        return data;

    } catch (error) {
        console.error('Failed to fetch search results:', error);
        toast.error(`Search failed: ${error.message}`);
        return null;
    }
};
