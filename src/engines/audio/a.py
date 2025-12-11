class NoteSoundMaker(PrintHandler):
    def __init__(self, note, ):
        
    
    def qwer(self, outdata, frames, n_frames):
        ## if current frame range has onbeat timing, play note
        frame_distance_at_block_start = n_frames % samples_per_note
        frame_distance_at_block_finish = (n_frames + frames - 1) % samples_per_note
        if frame_distance_at_block_start > frame_distance_at_block_finish:
            ## find onbeat timing
            n_frames_on_next_note = n_frames // samples_per_note + 1
            remaining_frames = n_frames_on_next_note * samples_per_note
            ix_onbeat = remaining_frames - n_frames
            # frame_range = np.arange(n_frames, n_frames + frames) % samples_per_note
            # ix_onbeat = np.argmax(frame_range == 0)
            ## remaining check
            len_sound = sound_note.shape[0]
            remaining = ix_onbeat + len_sound - frames
            if remaining > 0:
                ## need next block to play ticking
                prtwl("Index: ", ix_onbeat, "Tick len: ", len_sound, "Remaining: ", remaining)
                outdata[ix_onbeat: ] = sound_note[: -remaining]
                info_remaining = remaining
                flag_remaining = True
            else:
                ## can play ticking in this block at all
                outdata[ix_onbeat: ix_onbeat + len_sound] = sound_note

        # Remaining
        if flag_remaining:
            ## if remaining
            if info_remaining > frames:
                ## still need next block
                outdata[: ] = sound_note[-info_remaining: - (info_remaining - frames)]
                info_remaining -= frames
            else:
                outdata[: info_remaining] = sound_note[-info_remaining: ]
                flag_remaining = False
                info_remaining = 0