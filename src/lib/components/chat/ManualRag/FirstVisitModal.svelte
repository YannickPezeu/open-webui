<script lang="ts">
	import { onMount } from 'svelte';
	import { marked } from 'marked';
	import DOMPurify from 'dompurify';
	import Modal from '$lib/components/common/Modal.svelte';
	import { FIRST_VISIT_MODAL } from '$lib/constants';

	const STORAGE_KEY = 'hasSeenFirstVisitModal';

	let show = false;

	onMount(() => {
		// Vérifier si l'utilisateur a déjà vu le modal
		const hasSeenModal = localStorage.getItem(STORAGE_KEY);
		
		// if (!hasSeenModal && FIRST_VISIT_MODAL.ENABLED) {
            if (FIRST_VISIT_MODAL.ENABLED) {
			// Petit délai pour éviter que ça s'affiche trop brutalement
			setTimeout(() => {
				show = true;
			}, 500);
		}
	});

	const handleClose = () => {
		// Marquer comme vu dans localStorage
		localStorage.setItem(STORAGE_KEY, 'true');
		show = false;
	};

	const renderMarkdown = (text: string): string => {
		if (!text) return '';
		const html = marked.parse(text);
		return DOMPurify.sanitize(html);
	};
</script>

<Modal size="md" bind:show backdrop="static">
	<div class="px-6 pt-6 pb-5">
		<!-- Titre -->
		<div class="text-2xl font-semibold text-gray-900 dark:text-gray-100 mb-4 text-center">
			{FIRST_VISIT_MODAL.TITLE}
		</div>

		<!-- Message en Markdown -->
		<div class="prose dark:prose-invert max-w-none text-sm mb-6">
			{@html renderMarkdown(FIRST_VISIT_MODAL.MESSAGE)}
		</div>

		<!-- Bouton -->
		<div class="flex justify-center">
			<button
				on:click={handleClose}
				class="px-6 py-2.5 bg-gray-900 hover:bg-gray-800 dark:bg-white dark:hover:bg-gray-100 text-white dark:text-gray-900 rounded-lg font-medium transition-colors"
			>
				{FIRST_VISIT_MODAL.BUTTON_TEXT}
			</button>
		</div>
	</div>
</Modal>

<style>
	/* Style personnalisé pour le modal */
	:global(.modal-backdrop) {
		backdrop-filter: blur(4px);
	}
</style>