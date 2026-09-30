const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function setup({confirmed = true, fetcher} = {}) {
  const calls = [];
  const button = {disabled: false};
  const status = {textContent: ''};
  const dialog = {
    returnValue: '', listeners: new Map(),
    showModal() { this.open = true; },
    addEventListener(type, listener) { this.listeners.set(type, listener); },
    removeEventListener(type) { this.listeners.delete(type); },
    close(value) { this.returnValue = value; this.open = false; this.listeners.get('close')?.(); },
  };
  const elements = {stopConfirmDialog: dialog, stopConfirmTitle: {textContent: ''},
    stopConfirmDescription: {textContent: ''}, stopKeepRunningBtn: {focus() {}}};
  const context = vm.createContext({window: {confirm: () => confirmed},
    document: {getElementById: id => elements[id]},
    fetch: (...args) => { calls.push(args); return fetcher(...args); }});
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/stop-confirm.js'), 'utf8'), context);
  return {stop: context.window.StudioStop, dialog, button, status, calls};
}

test('closing confirmation keeps the job running without a stop request', async () => {
  const ui = setup({fetcher: async () => ({ok: true})});
  const pending = ui.stop.request({jobId: 'job_123', url: '/api/stop/job_123', button: ui.button, statusElement: ui.status});
  ui.dialog.close('keep');
  const result = await pending;
  assert.equal(result.confirmed, false);
  assert.equal(ui.calls.length, 0);
  assert.equal(ui.button.disabled, false);
});

test('confirmed stop sends exactly one request and only acknowledges the request', async () => {
  let release;
  const ui = setup({fetcher: () => new Promise(resolve => { release = resolve; })});
  const pending = ui.stop.request({jobId: 'job_123', url: '/api/stop/job_123', button: ui.button, statusElement: ui.status});
  ui.dialog.close('stop');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(ui.button.disabled, true);
  const duplicate = await ui.stop.request({jobId: 'job_123', url: '/api/stop/job_123', button: ui.button, statusElement: ui.status});
  assert.equal(duplicate.ok, false);
  assert.equal(ui.calls.length, 1);
  release({ok: true, json: async () => ({status: 'ok', cancel_requested: true})});
  const result = await pending;
  assert.equal(result.ok, true);
  assert.match(ui.status.textContent, /yêu cầu dừng/i);
  assert.doesNotMatch(ui.status.textContent, /đã dừng hoàn toàn/i);
});

test('failed request restores stop button and says processing may continue', async () => {
  const ui = setup({fetcher: async () => ({ok: false, json: async () => ({error: 'Server busy'})})});
  const pending = ui.stop.request({jobId: 'job_123', url: '/api/stop/job_123', button: ui.button, statusElement: ui.status});
  ui.dialog.close('stop');
  const result = await pending;
  assert.equal(result.ok, false);
  assert.equal(ui.button.disabled, false);
  assert.match(ui.status.textContent, /vẫn đang chạy/i);
});
