/**
 * ui/tts-ref-controller.js — TTS reference audio controller
 *
 * Creates a parameterized TTS ref audio controller for duplex pages.
 * Depends on global RefAudioPlayer (loaded via regular <script> tag).
 */

const RefAudioPlayer = window.RefAudioPlayer;

/**
 * Creates a TTS ref audio controller for a given page prefix.
 * @param {string} prefix - Element ID prefix (e.g. 'omni' or 'duplex')
 * @param {function} getRefAudioBase64 - Function returning current LLM ref audio base64
 * @returns {{ mode: string, audioBase64: string|null, rap: object|null,
 *             onModeChange: function, updateHint: function, getBase64: function, init: function }}
 */
export function createTtsRefController(prefix, getRefAudioBase64) {
    const ctrl = {
        mode: 'extract',
        audioBase64: null,
        rap: null,

        init() {
            const uploadEl = document.getElementById(`${prefix}TtsRefUpload`);
            if (!uploadEl) return;
            ctrl.rap = new RefAudioPlayer(uploadEl, {
                theme: 'light',
                onUpload(base64, name, duration) {
                    ctrl.audioBase64 = base64;
                    ctrl.updateHint();
                },
                onRemove() {
                    ctrl.audioBase64 = null;
                    if (ctrl.rap) ctrl.rap.clear();
                    ctrl.updateHint();
                },
            });
            ctrl.updateHint();
        },

        onModeChange() {
            const radio = document.querySelector(`input[name="${prefix}TtsRefMode"]:checked`);
            ctrl.mode = radio?.value || 'extract';
            const uploadEl = document.getElementById(`${prefix}TtsRefUpload`);
            if (uploadEl) uploadEl.style.display = ctrl.mode === 'independent' ? '' : 'none';
            ctrl.updateHint();
        },

        updateHint() {
            const el = document.getElementById(`${prefix}TtsRefHint`);
            if (!el) return;
            if (ctrl.mode === 'extract') {
                if (getRefAudioBase64()) {
                    el.textContent = 'Will use the same audio as LLM Ref Audio';
                    el.style.color = '#aaa';
                } else {
                    el.textContent = 'No LLM Ref Audio loaded \u2014 please upload independently';
                    el.style.color = '#d4a017';
                }
            } else {
                if (ctrl.audioBase64) {
                    el.textContent = 'Using independent TTS Ref Audio';
                    el.style.color = '#aaa';
                } else {
                    el.textContent = 'Please upload a TTS reference audio';
                    el.style.color = '#d4a017';
                }
            }
        },

        getBase64() {
            if (ctrl.mode === 'extract') {
                return getRefAudioBase64();
            }
            return ctrl.audioBase64;
        },
    };
    return ctrl;
}
