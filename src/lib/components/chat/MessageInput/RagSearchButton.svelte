<script lang="ts">
	import { DropdownMenu } from 'bits-ui';
	import { flyAndScale } from '$lib/utils/transitions';
	import { getContext } from 'svelte';
	import Search from '../../icons/Search.svelte';
	import Dropdown from '../../common/Dropdown.svelte';
	import Tooltip from '../../common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let ragSearchEnabled = false;
	export let ragExpertMode = false;
	export let onClose: Function = () => {};

	let show = false;
</script>

<div class="flex items-center gap-0.5">
	<!-- Bouton principal toggle ON/OFF -->
	<Tooltip content={ragSearchEnabled ? $i18n.t('RAG search enabled') : $i18n.t('RAG search disabled')} placement="top">
		<button
			on:click|preventDefault={() => {
				ragSearchEnabled = !ragSearchEnabled;
				if (!ragSearchEnabled) {
					ragExpertMode = false;
					show = false; // Ferme le dropdown si on désactive
				}
			}}
			type="button"
			class="px-2 @xl:pl-2.5 @xl:pr-1.5 py-2 flex gap-1.5 items-center text-sm rounded-l-full transition-colors duration-300 focus:outline-hidden overflow-hidden hover:bg-gray-50 dark:hover:bg-gray-800 {ragSearchEnabled
				? 'text-sky-500 dark:text-sky-300 bg-sky-50 dark:bg-sky-200/5'
				: 'bg-transparent text-gray-600 dark:text-gray-300'} {!ragSearchEnabled ? 'rounded-r-full' : ''}"
		>
			<Search className="size-4" strokeWidth="1.75" />
			<span
				class="hidden @xl:block whitespace-nowrap overflow-hidden text-ellipsis leading-none"
			>
				{$i18n.t('RAG Search')}
			</span>
		</button>
	</Tooltip>

	<!-- Bouton dropdown pour les options (visible seulement si activé) -->
	{#if ragSearchEnabled}
		<Dropdown
			bind:show
			on:change={(e) => {
				if (e.detail === false) {
					onClose();
				}
			}}
		>
			<Tooltip content={$i18n.t('RAG options')} placement="top">
				<button
					type="button"
					class="pr-2 @xl:pr-2.5 py-2 flex items-center text-sm rounded-r-full transition-colors duration-300 focus:outline-hidden hover:bg-gray-50 dark:hover:bg-gray-800 text-sky-500 dark:text-sky-300 bg-sky-50 dark:bg-sky-200/5"
				>
					<svg 
						xmlns="http://www.w3.org/2000/svg" 
						viewBox="0 0 20 20" 
						fill="currentColor" 
						class="size-4"
					>
						<path fill-rule="evenodd" d="M5.23 7.21a.75.75 0 011.06.02L10 11.168l3.71-3.938a.75.75 0 111.08 1.04l-4.25 4.5a.75.75 0 01-1.08 0l-4.25-4.5a.75.75 0 01.02-1.06z" clip-rule="evenodd" />
					</svg>
				</button>
			</Tooltip>

			<div slot="content">
				<DropdownMenu.Content
					class="w-full max-w-[220px] rounded-xl px-1 py-1 border-gray-300/30 dark:border-gray-700/50 z-50 bg-white dark:bg-gray-850 dark:text-white shadow-xl"
					sideOffset={8}
					side="top"
					align="center"
					transition={flyAndScale}
				>
					<label class="flex items-center gap-2.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 rounded-lg p-2.5 transition">
						<input
							type="checkbox"
							bind:checked={ragExpertMode}
							on:click|stopPropagation
							class="size-4 rounded border-gray-300 text-sky-600 focus:ring-sky-500 flex-shrink-0"
						/>
						<div class="flex flex-col gap-0.5">
							<span class="text-sm font-medium text-gray-900 dark:text-white whitespace-nowrap">
								{$i18n.t('Expert Mode')}
							</span>
							<span class="text-xs text-gray-500 dark:text-gray-400 leading-tight">
								{$i18n.t('Manually select sources')}
							</span>
						</div>
					</label>
				</DropdownMenu.Content>
			</div>
		</Dropdown>
	{/if}
</div>