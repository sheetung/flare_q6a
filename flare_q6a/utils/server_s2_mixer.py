import torch


class QuadXMixer:
    def __init__(self, roll_p, roll_i, roll_d, pitch_p, pitch_i, pitch_d, yaw_p, yaw_i, yaw_d,
                 i_limit=None, d_alpha=None, device='cuda'):
        self.roll_p = roll_p
        self.roll_i = roll_i
        self.roll_d = roll_d
        self.pitch_p = pitch_p
        self.pitch_i = pitch_i
        self.pitch_d = pitch_d
        self.yaw_p = yaw_p
        self.yaw_i = yaw_i
        self.yaw_d = yaw_d

        self.i_limit = i_limit
        self.d_alpha = d_alpha

        self.roll_vel_err_sum = 0
        self.pitch_vel_err_sum = 0
        self.yaw_vel_err_sum = 0

        self.last_roll_vel_err = 0
        self.last_pitch_vel_err = 0
        self.last_yaw_vel_err = 0

        self.roll_d_filtered = 0
        self.pitch_d_filtered = 0
        self.yaw_d_filtered = 0

        self.mixer_matrix = torch.tensor([
            [1.0,  -1.0,  -1.0,  -1.0],
            [1.0,  -1.0,   1.0,   1.0],
            [1.0,   1.0,   1.0,  -1.0],
            [1.0,   1.0,  -1.0,   1.0],
        ], device=device)

    def reset(self, envs_idx=None):
        states = [
            "roll_vel_err_sum",
            "pitch_vel_err_sum",
            "yaw_vel_err_sum",
            "last_roll_vel_err",
            "last_pitch_vel_err",
            "last_yaw_vel_err",
            "roll_d_filtered",
            "pitch_d_filtered",
            "yaw_d_filtered",
        ]
        for name in states:
            value = getattr(self, name)
            if torch.is_tensor(value):
                if envs_idx is None:
                    value.zero_()
                else:
                    value[envs_idx] = 0
            else:
                setattr(self, name, 0)

    def _clamp_err_sum(self, err_sum):
        if self.i_limit is not None:
            return torch.clamp(err_sum, -self.i_limit, self.i_limit)
        return err_sum

    def _apply_d_filter(self, d_raw, d_filtered_prev):
        if self.d_alpha is not None:
            return self.d_alpha * d_raw + (1.0 - self.d_alpha) * d_filtered_prev
        return d_raw

    def pid_attitude_command_for_mix(self, roll_vel_r, pitch_vel_r, yaw_vel_r, roll_vel_d, pitch_vel_d, yaw_vel_d, dt):
        '''
            input:
                *_vel_r: real angular velocity
                *_vel_d: desired angular velocity
                dt: time interval
            output: motor_speed: attitude command for mix
        '''
        roll_vel_err = roll_vel_d - roll_vel_r
        pitch_vel_err = pitch_vel_d - pitch_vel_r
        yaw_vel_err = yaw_vel_d - yaw_vel_r

        self.roll_vel_err_sum = self._clamp_err_sum(self.roll_vel_err_sum + roll_vel_err)
        self.pitch_vel_err_sum = self._clamp_err_sum(self.pitch_vel_err_sum + pitch_vel_err)
        self.yaw_vel_err_sum = self._clamp_err_sum(self.yaw_vel_err_sum + yaw_vel_err)

        roll_d_raw = self.roll_d * (roll_vel_err - self.last_roll_vel_err) / dt
        pitch_d_raw = self.pitch_d * (pitch_vel_err - self.last_pitch_vel_err) / dt
        yaw_d_raw = self.yaw_d * (yaw_vel_err - self.last_yaw_vel_err) / dt

        self.roll_d_filtered = self._apply_d_filter(roll_d_raw, self.roll_d_filtered)
        self.pitch_d_filtered = self._apply_d_filter(pitch_d_raw, self.pitch_d_filtered)
        self.yaw_d_filtered = self._apply_d_filter(yaw_d_raw, self.yaw_d_filtered)

        roll_pid = self.roll_p * roll_vel_err + self.roll_i * self.roll_vel_err_sum * dt + self.roll_d_filtered
        pitch_pid = self.pitch_p * pitch_vel_err + self.pitch_i * self.pitch_vel_err_sum * dt + self.pitch_d_filtered
        yaw_pid = self.yaw_p * yaw_vel_err + self.yaw_i * self.yaw_vel_err_sum * dt + self.yaw_d_filtered

        self.last_roll_vel_err = roll_vel_err
        self.last_pitch_vel_err = pitch_vel_err
        self.last_yaw_vel_err = yaw_vel_err

        return roll_pid, pitch_pid, yaw_pid

    def mix(self, throttle: float, roll_pid: float, pitch_pid: float, yaw_pid: float):
        '''
            input:
                throttle: throttle value
                *_pid: 3 axis pid command value
            output:
                motor_speed: motor speed for 4 motors
        '''
        controls = torch.stack([throttle, roll_pid, pitch_pid, yaw_pid])
        motor_speed = torch.matmul(self.mixer_matrix, controls)
        motor_speed = motor_speed.t()

        return torch.clamp(motor_speed, 0, 1)
