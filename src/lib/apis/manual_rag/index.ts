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
    toast.info('Searching...');

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