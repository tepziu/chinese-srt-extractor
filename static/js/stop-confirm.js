/* Shared stop confirmation and request lifecycle for Studio, TTS and batch jobs. */
(function () {
  'use strict';

  const pending = new Set();
  let asking = false;

  function ask(title, description) {
    if (asking) return Promise.resolve(false);
    const dialog = document.getElementById('stopConfirmDialog');
    if (!dialog || typeof dialog.showModal !== 'function') {
      return Promise.resolve(window.confirm(description));
    }
    asking = true;
    document.getElementById('stopConfirmTitle').textContent = title;
    document.getElementById('stopConfirmDescription').textContent = description;
    dialog.returnValue = 'keep';
    return new Promise(resolve => {
      const onClose = () => {
        dialog.removeEventListener('close', onClose);
        asking = false;
        resolve(dialog.returnValue === 'stop');
      };
      dialog.addEventListener('close', onClose);
      dialog.showModal();
      document.getElementById('stopKeepRunningBtn')?.focus();
    });
  }

  async function request({jobId, url, button, statusElement, title = 'Dừng công việc này?',
                          description = 'Bạn có chắc muốn dừng? Các tệp đã hoàn tất vẫn được giữ lại.'}) {
    if (!jobId || pending.has(jobId)) return {ok: false, confirmed: false};
    const confirmed = await ask(title, description);
    if (!confirmed || pending.has(jobId)) return {ok: false, confirmed: false};
    pending.add(jobId);
    if (button) button.disabled = true;
    if (statusElement) statusElement.textContent = 'Đang gửi yêu cầu dừng…';
    try {
      const response = await fetch(url, {method: 'POST'});
      const data = await response.json().catch(() => ({}));
      if (!response.ok || data.error) throw new Error(data.error || `HTTP ${response.status}`);
      if (statusElement) statusElement.textContent =
        'Đã gửi yêu cầu dừng. Công đoạn hiện tại có thể cần thời gian để kết thúc.';
      return {ok: true, confirmed: true, data};
    } catch (error) {
      if (button) button.disabled = false;
      if (statusElement) statusElement.textContent =
        `Không gửi được yêu cầu dừng: ${error.message}. Công việc có thể vẫn đang chạy; hãy kiểm tra trạng thái.`;
      return {ok: false, confirmed: true, error};
    } finally {
      pending.delete(jobId);
    }
  }

  window.StudioStop = {ask, request};
})();
