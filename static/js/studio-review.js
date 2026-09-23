/* Full SRT editor. Polling never replaces text after it has been loaded. */
window.StudioReviewEditor = (() => {
    let jobId = null, revision = 0, loading = null, timer = null;
    let saves = Promise.resolve();
    let recoveryDraft = null;
    const field = () => document.getElementById('reviewSrtTextarea');
    const storageKey = id => `studio-review:${id}`;
    function showError(error) {
        const status = document.getElementById('reviewSaveStatus');
        if (status) status.textContent = error.message;
    }
    function save() {
        clearTimeout(timer);
        const id = jobId, content = field().value;
        saves = saves.catch(() => {}).then(async () => {
            if (jobId !== id) return;
            const response = await fetch(`/api/pipeline/review/${id}`, {
                method: 'PUT', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({content, revision})
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Chưa lưu được bản nháp');
            revision = data.revision;
            const status = document.getElementById('reviewSaveStatus');
            if (status) status.textContent = 'Đã lưu bản nháp';
        });
        saves.catch(showError);
        return saves;
    }
    async function load(id) {
        if (jobId === id) return loading;
        jobId = id;
        loading = (async () => {
            const textarea = field();
            textarea.disabled = true;
            try {
                const response = await fetch(`/api/pipeline/review/${id}`);
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Không tải được toàn bộ SRT');
                if (jobId !== id) return;
                revision = data.revision;
                textarea.value = data.content;
                textarea.oninput = () => {
                    try { localStorage.setItem(storageKey(id), JSON.stringify({content: textarea.value, revision})); } catch (_) {}
                    clearTimeout(timer);
                    timer = setTimeout(save, 700);
                };
                try {
                    const draft = JSON.parse(localStorage.getItem(storageKey(id)) || 'null');
                    if (draft && draft.revision === revision) textarea.value = draft.content;
                    else if (draft && draft.content !== data.content) {
                        recoveryDraft = draft.content;
                        showError(new Error('Có bản nháp trên máy khác phiên bản server. Có thể khôi phục bằng nút bên dưới.'));
                        const button = document.getElementById('reviewRestoreLocal');
                        if (button) button.hidden = false;
                    }
                } catch (_) {}
            } catch (error) {
                jobId = null;
                showError(error);
                throw error;
            } finally { textarea.disabled = jobId !== id; }
        })();
        return loading;
    }
    return {
        load,
        restoreLocalDraft() {
            if (recoveryDraft === null) return;
            field().value = recoveryDraft;
            field().oninput();
            recoveryDraft = null;
            const button = document.getElementById('reviewRestoreLocal');
            if (button) button.hidden = true;
        },
        async submitData() {
            await loading;
            await save();
            return {updated_srt: field().value, revision};
        },
        complete() {
            clearTimeout(timer);
            try { localStorage.removeItem(storageKey(jobId)); } catch (_) {}
            jobId = null;
            recoveryDraft = null;
        }
    };
})();
