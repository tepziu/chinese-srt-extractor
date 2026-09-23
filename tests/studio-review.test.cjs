const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function setup(fetch, savedDraft = null) {
    const textarea = {value: '', disabled: false}, status = {};
    const context = vm.createContext({window: {}, fetch, setTimeout, clearTimeout,
        document: {getElementById: id => id === 'reviewSrtTextarea' ? textarea : status},
        localStorage: {getItem: () => savedDraft && JSON.stringify(savedDraft), setItem() {}, removeItem() {}}});
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/studio-review.js'), 'utf8'), context);
    return {editor: context.window.StudioReviewEditor, textarea, status};
}
test('polling preserves typed text and submit flushes complete draft', async () => {
    const requests = [];
    const text = 'Complete subtitle '.repeat(200);
    const {editor, textarea} = setup(async (url, options) => {
        requests.push(options);
        return {ok: true, json: async () => options ? {revision: 5} : {revision: 4, content: text}};
    });
    await editor.load('review01');
    assert.equal(textarea.value, text);
    textarea.value = text + ' edited';
    await editor.load('review01');
    assert.equal(requests.length, 1);
    assert.equal(textarea.value, text + ' edited');
    const payload = await editor.submitData();
    assert.equal(payload.revision, 5);
    assert.equal(payload.updated_srt, text + ' edited');
    assert.equal(JSON.parse(requests[1].body).revision, 4);
    editor.complete();
});
test('revision conflict blocks submit and keeps text', async () => {
    const {editor, textarea} = setup(async (url, options) => ({ok: !options,
        json: async () => options ? {error: 'Revision conflict'} : {revision: 1, content: 'Full SRT'}}));
    await editor.load('review02');
    textarea.value = 'My unsaved work';
    await assert.rejects(editor.submitData(), /Revision conflict/);
    assert.equal(textarea.value, 'My unsaved work');
    editor.complete();
});
test('older local draft can be explicitly recovered after reload', async () => {
    const {editor, textarea} = setup(async (url, options) => ({ok: true,
        json: async () => options ? {revision: 4} : {revision: 3, content: 'Server version'}}),
        {revision: 2, content: 'Recovered local edit'});
    await editor.load('review03');
    assert.equal(textarea.value, 'Server version');
    editor.restoreLocalDraft();
    assert.equal(textarea.value, 'Recovered local edit');
    const data = await editor.submitData();
    assert.equal(data.revision, 4);
    assert.equal(data.updated_srt, 'Recovered local edit');
    editor.complete();
});
