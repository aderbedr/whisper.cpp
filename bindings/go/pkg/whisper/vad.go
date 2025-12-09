package whisper

import (
	"fmt"
	"os"
	"runtime"
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

type vad struct {
	ctx  *whisper.VADContext
	path string
}

// Make sure vad adheres to the interface
var _ VAD = (*vad)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

// NewVAD creates a new VAD instance from model file with default parameters
func NewVAD(modelPath string) (VAD, error) {
	return NewVADWithParams(modelPath, runtime.NumCPU(), false, 0)
}

// NewVADWithParams creates a new VAD instance with custom parameters
func NewVADWithParams(modelPath string, nThreads int, useGPU bool, gpuDevice int) (VAD, error) {
	if _, err := os.Stat(modelPath); err != nil {
		return nil, fmt.Errorf("VAD model file not found: %w", err)
	}

	ctx := whisper.Whisper_vad_init_from_file_with_params(modelPath, nThreads, useGPU, gpuDevice)
	if ctx == nil {
		return nil, fmt.Errorf("failed to initialize VAD context from file: %s", modelPath)
	}

	return &vad{
		ctx:  ctx,
		path: modelPath,
	}, nil
}

// Close releases VAD resources
func (v *vad) Close() error {
	if v.ctx != nil {
		v.ctx.Whisper_vad_free()
		v.ctx = nil
	}
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (v *vad) String() string {
	str := "<whisper.vad"
	if v.ctx != nil {
		str += fmt.Sprintf(" model=%q", v.path)
	}
	return str + ">"
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// DetectSpeech returns true if speech is detected in the audio samples
func (v *vad) DetectSpeech(samples []float32) bool {
	if v.ctx == nil {
		return false
	}
	return v.ctx.Whisper_vad_detect_speech(samples)
}

// GetProbabilities returns the current VAD probabilities
func (v *vad) GetProbabilities() []float32 {
	if v.ctx == nil {
		return nil
	}
	return v.ctx.Whisper_vad_probs()
}

// SegmentFromSamples segments audio based on speech detection
func (v *vad) SegmentFromSamples(samples []float32) ([]VADSegment, error) {
	if v.ctx == nil {
		return nil, ErrInternalAppError
	}

	// Use default VAD parameters for now
	vadParams := whisper.Whisper_vad_default_params()

	// Get segments from samples
	segments := v.ctx.Whisper_vad_segments_from_samples(vadParams, samples)
	if segments == nil {
		return nil, fmt.Errorf("failed to segment audio samples")
	}
	defer segments.Whisper_vad_free_segments()

	return convertVADSegments(segments), nil
}

// SegmentFromProbabilities segments audio using pre-computed probabilities
func (v *vad) SegmentFromProbabilities() ([]VADSegment, error) {
	if v.ctx == nil {
		return nil, ErrInternalAppError
	}

	// Use default VAD parameters for now
	vadParams := whisper.Whisper_vad_default_params()

	// Get segments from probabilities
	segments := v.ctx.Whisper_vad_segments_from_probs(vadParams)
	if segments == nil {
		return nil, fmt.Errorf("failed to segment from probabilities")
	}
	defer segments.Whisper_vad_free_segments()

	return convertVADSegments(segments), nil
}

///////////////////////////////////////////////////////////////////////////////
// UTILITY FUNCTIONS

// SegmentAudioBySilence is a convenience function to segment audio by detecting silences
func SegmentAudioBySilence(modelPath string, samples []float32, silenceThreshold float32) ([]VADSegment, error) {
	vad, err := NewVAD(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create VAD: %w", err)
	}
	defer vad.Close()

	return vad.SegmentFromSamples(samples)
}

// DetectSilences returns segments where silence is detected
func DetectSilences(segments []VADSegment) []VADSegment {
	var silences []VADSegment
	for _, segment := range segments {
		if !segment.IsSpeech {
			silences = append(silences, segment)
		}
	}
	return silences
}

// DetectSpeechSegments returns segments where speech is detected
func DetectSpeechSegments(segments []VADSegment) []VADSegment {
	var speechSegments []VADSegment
	for _, segment := range segments {
		if segment.IsSpeech {
			speechSegments = append(speechSegments, segment)
		}
	}
	return speechSegments
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

// convertVADSegments converts C VAD segments to Go VADSegment structs
func convertVADSegments(segments *whisper.VADSegments) []VADSegment {
	if segments == nil {
		return nil
	}

	n := segments.Whisper_vad_segments_n_segments()
	result := make([]VADSegment, n)

	for i := 0; i < n; i++ {
		start := segments.Whisper_vad_segments_get_segment_t0(i)
		end := segments.Whisper_vad_segments_get_segment_t1(i)

		result[i] = VADSegment{
			Start:    time.Duration(start * float32(time.Second)),
			End:      time.Duration(end * float32(time.Second)),
			IsSpeech: true, // Assuming returned segments are speech segments
		}
	}

	return result
}
