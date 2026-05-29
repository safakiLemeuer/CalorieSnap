import React, { useState, useRef } from 'react';
import {
  StyleSheet, View, Text, TouchableOpacity, Image,
  ActivityIndicator, Alert, Dimensions, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system';
import { StatusBar } from 'expo-status-bar';

const API = 'http://192.168.1.252:8020';
const { width: SW } = Dimensions.get('window');

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<any>(null);
  const [photoUri, setPhotoUri] = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);
  const [calories, setCalories] = useState<number | null>(null);
  const [note, setNote]         = useState('');

  if (!permission) {
    return <View style={S.container} />;
  }

  if (!permission.granted) {
    return (
      <View style={S.container}>
        <Text style={S.permText}>Camera permission needed</Text>
        <TouchableOpacity style={S.btn} onPress={requestPermission}>
          <Text style={S.btnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  async function takePhoto() {
    try {
      const photo = await cameraRef.current?.takePictureAsync({ quality: 0.7 });
      if (photo?.uri) {
        setPhotoUri(photo.uri);
        setCalories(null);
      }
    } catch (e: any) {
      Alert.alert('Camera error', e.message);
    }
  }

  async function pickPhoto() {
    const res = await ImagePicker.launchImageLibraryAsync({ quality: 0.7 });
    if (!res.canceled && res.assets[0]) {
      setPhotoUri(res.assets[0].uri);
      setCalories(null);
    }
  }

  async function analyze() {
    if (!photoUri) return;
    setLoading(true);
    try {
      const b64 = await FileSystem.readAsStringAsync(photoUri, {
        encoding: FileSystem.EncodingType.Base64,
      });
      const r = await fetch(`${API}/snap`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ image_base64: b64, hint: '' }),
      });
      if (!r.ok) throw new Error('Server error ' + r.status);
      const d = await r.json();
      setCalories(d.total_calories);
      setNote(d.note || '');
    } catch (e: any) {
      Alert.alert('Error', e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={S.container}>
      <StatusBar style="light" />
      <Text style={S.title}>CalorieSnap</Text>
      <Text style={S.sub}>BHTLabs · Air-gapped · Moondream</Text>

      <View style={S.camBox}>
        {photoUri ? (
          <Image source={{ uri: photoUri }} style={S.camView} resizeMode="cover" />
        ) : (
          <CameraView ref={cameraRef} style={S.camView} facing="back" />
        )}
      </View>

      {calories !== null && (
        <View style={S.resultBox}>
          <Text style={S.calNum}>{calories}</Text>
          <Text style={S.calLabel}>calories</Text>
          {!!note && <Text style={S.calNote}>{note}</Text>}
        </View>
      )}

      <View style={S.btnRow}>
        {photoUri ? (
          <>
            <TouchableOpacity
              style={S.secBtn}
              onPress={() => { setPhotoUri(null); setCalories(null); }}
            >
              <Text style={S.secBtnText}>Retake</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[S.btn, loading && S.btnDim]}
              onPress={analyze}
              disabled={loading}
            >
              {loading
                ? <ActivityIndicator color="#fff" />
                : <Text style={S.btnText}>Analyze 🔍</Text>}
            </TouchableOpacity>
          </>
        ) : (
          <>
            <TouchableOpacity style={S.secBtn} onPress={pickPhoto}>
              <Text style={S.secBtnText}>Library 📷</Text>
            </TouchableOpacity>
            <TouchableOpacity style={S.btn} onPress={takePhoto}>
              <Text style={S.btnText}>Snap 📸</Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      <Text style={S.footer}>🔒 On-device · Zero cloud · BHTLabs</Text>
    </View>
  );
}

const S = StyleSheet.create({
  container:  { flex: 1, backgroundColor: '#080a0c', paddingTop: Platform.OS === 'ios' ? 60 : 20 },
  title:      { color: '#fff', fontSize: 26, fontWeight: '800', textAlign: 'center', letterSpacing: -0.5 },
  sub:        { color: '#22c55e', fontSize: 12, textAlign: 'center', marginBottom: 12 },
  camBox:     { flex: 1, marginHorizontal: 16, borderRadius: 16, overflow: 'hidden', backgroundColor: '#111' },
  camView:    { flex: 1 },
  resultBox:  { alignItems: 'center', paddingVertical: 18 },
  calNum:     { fontSize: 72, fontWeight: '800', color: '#22c55e', lineHeight: 76 },
  calLabel:   { fontSize: 16, color: '#6b7280', marginTop: -4 },
  calNote:    { fontSize: 12, color: '#4b5563', marginTop: 6, textAlign: 'center', paddingHorizontal: 20 },
  btnRow:     { flexDirection: 'row', gap: 12, padding: 16, paddingBottom: Platform.OS === 'ios' ? 34 : 16 },
  btn:        { flex: 2, backgroundColor: '#22c55e', borderRadius: 14, paddingVertical: 16, alignItems: 'center' },
  btnDim:     { backgroundColor: '#166534' },
  btnText:    { color: '#fff', fontSize: 17, fontWeight: '700' },
  secBtn:     { flex: 1, backgroundColor: '#111', borderRadius: 14, paddingVertical: 16, alignItems: 'center', borderWidth: 1, borderColor: '#1c2128' },
  secBtnText: { color: '#6b7280', fontSize: 15 },
  footer:     { color: '#1f2937', fontSize: 11, textAlign: 'center', paddingBottom: 8 },
  permText:   { color: '#fff', textAlign: 'center', fontSize: 16, marginTop: 100, marginBottom: 24 },
});
